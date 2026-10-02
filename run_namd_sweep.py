#!/usr/bin/env python3
"""Batch NAMD runner: sweep temperature / minimization / MD-step settings.

Takes an existing NAMD configuration file as a template, writes one patched
copy per parameter set into its own results directory, runs NAMD on each, and
collects the final energies into results/summary.csv.  No VMD or NAMDgui
needed.

Examples
--------
  # Preview the matrix and total step count without running anything
  ./run_namd_sweep.py --temperatures 300 400 500 --minimize 1000 --md 10000 --estimate

  # 3 temperatures x 2 MD lengths, 4 NAMD threads each, 2 jobs at a time
  ./run_namd_sweep.py --temperatures 300 400 500 --minimize 1000 --md 10000 50000 \
      --threads 4 --jobs 2

  # Paired scenarios (zip) instead of the full cartesian product
  ./run_namd_sweep.py --mode zip --temperatures 300 400 500 \
      --minimize 1000 2000 5000 --md 10000 20000 30000

  # Python-range syntax start:stop:step (stop exclusive)
  ./run_namd_sweep.py --temperatures 300:3500:500 --minimize 1000 --md 10000:100000:10000

  # Rebuild summary.csv from runs that already finished
  ./run_namd_sweep.py --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# NAMD config file editing
# ---------------------------------------------------------------------------

# Directives whose value is a path NAMD reads -> must be absolute once the
# config file is moved into a per-run subdirectory.
INPUT_PATH_KEYS = (
    "coordinates", "structure", "parameters", "velocities", "bincoordinates",
    "binvelocities", "extendedSystem", "consref", "conskfile", "fixedAtomsFile",
    "extraBondsFile", "parmfile", "ambercoor", "grofile", "topfile",
    "colvarsConfig", "colvarsInput", "tclForcesScript",
)

# Directives whose value is a path NAMD writes.  In the NAMDgui template these
# are all derived from `$output`, so they are only rewritten when literal.
OUTPUT_PATH_KEYS = (
    "outputname", "dcdfile", "xstfile", "restartname", "veldcdfile",
    "forcedcdfile", "DCDfile",
)


def _directive_re(key: str) -> re.Pattern:
    """Match `key value [trailing]`, allowing multi-word keys like `set output`."""
    body = r"\s+".join(re.escape(p) for p in key.split())
    return re.compile(rf"^(\s*)({body})(\s+)(\S+)(.*)$", re.IGNORECASE)


def get_directive(lines: list[str], key: str, default=None):
    rx = _directive_re(key)
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        m = rx.match(line)
        if m:
            return m.group(4)
    return default


def map_directive(lines: list[str], key: str, fn) -> int:
    """Rewrite every occurrence of `key`'s value through `fn`. Returns the count.

    Column alignment is preserved by keeping the original gap, so the patched
    file still reads like the hand-written one.
    """
    rx = _directive_re(key)
    hits = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        m = rx.match(line)
        if not m:
            continue
        indent, keytext, gap, old, rest = m.groups()
        new = fn(old)
        if new is None:
            continue
        lines[i] = f"{indent}{keytext}{gap}{new}{rest}"
        hits += 1
    return hits


# Tcl commands that consume the configuration; anything added later is ignored.
ACTION_KEYS = ("minimize", "run", "reinitvels", "numsteps", "runcontroller")


def _action_index(lines: list[str]) -> int:
    """Index of the first Tcl action line, i.e. where the config section ends."""
    rxs = [_directive_re(k) for k in ACTION_KEYS]
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        if any(rx.match(line) for rx in rxs):
            return i
    return len(lines)


def set_directive(lines: list[str], key: str, value, *, add_if_missing=True) -> int:
    hits = map_directive(lines, key, lambda _old: str(value))
    if hits == 0 and add_if_missing:
        # Insert before minimize/run, otherwise NAMD never sees the setting.
        at = _action_index(lines)
        if key.split()[0].lower() in ACTION_KEYS:
            at = len(lines)
        lines.insert(at, f"{key:<24}{value}")
        hits = 1
    return hits


def comment_out(lines: list[str], key: str, note: str = "") -> int:
    rx = _directive_re(key)
    hits = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        if rx.match(line):
            suffix = f"   # {note}" if note else ""
            lines[i] = f"# {line}{suffix}"
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def _num(tok):
    f = float(tok)
    return int(f) if f.is_integer() else f


def parse_numlist(tokens) -> list:
    """Parse numbers, comma lists and `start:stop[:step]` ranges (stop exclusive)."""
    out = []
    for tok in tokens or []:
        for piece in str(tok).split(","):
            piece = piece.strip()
            if not piece:
                continue
            if ":" in piece:
                bits = [b.strip() for b in piece.split(":")]
                if len(bits) == 2:
                    start, stop, step = _num(bits[0]), _num(bits[1]), 1
                elif len(bits) == 3:
                    start, stop, step = (_num(b) for b in bits)
                else:
                    raise ValueError(f"bad range spec: {piece!r}")
                if step == 0:
                    raise ValueError(f"zero step in range spec: {piece!r}")
                n = 0
                cur = start
                while (step > 0 and cur < stop) or (step < 0 and cur > stop):
                    out.append(_num(f"{cur:.10g}"))
                    n += 1
                    cur = start + n * step
            else:
                out.append(_num(piece))
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


@dataclass(frozen=True)
class SimSpec:
    temperature: float
    minimize_steps: int
    run_steps: int
    label: str | None = None

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        t = f"{self.temperature:g}".replace(".", "p").replace("-", "m")
        return f"T{t}_min{self.minimize_steps}_md{self.run_steps}"


def build_specs(temps, mins, mds, mode: str) -> list[SimSpec]:
    if mode == "product":
        combos = itertools.product(temps, mins, mds)
    elif mode == "zip":
        n = max(len(temps), len(mins), len(mds))
        pad = lambda xs: list(xs) + [xs[-1]] * (n - len(xs))
        combos = zip(pad(temps), pad(mins), pad(mds))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return [SimSpec(float(t), int(mn), int(md)) for t, mn, md in combos]


def specs_from_file(path: Path) -> tuple[list[SimSpec], str]:
    data = json.loads(Path(path).read_text())
    if "runs" in data:
        specs = [
            SimSpec(
                float(r["temperature"]),
                int(r.get("minimize", r.get("minimize_steps", 0))),
                int(r.get("md", r.get("run_steps", 0))),
                r.get("label"),
            )
            for r in data["runs"]
        ]
        return specs, "explicit"
    temps = parse_numlist(data.get("temperatures", []))
    mins = parse_numlist(data.get("minimize", []))
    mds = parse_numlist(data.get("md", []))
    return build_specs(temps, mins, mds, data.get("mode", "product")), data.get("mode", "product")


# ---------------------------------------------------------------------------
# Config rendering
# ---------------------------------------------------------------------------

def render_config(template_lines: list[str], spec: SimSpec, run_dir: Path,
                  base_dir: Path, overrides: dict[str, str]) -> str:
    lines = list(template_lines)

    absolutize = lambda v: v if v.startswith(("$", "/")) else str((base_dir / v).resolve())
    for key in INPUT_PATH_KEYS:
        map_directive(lines, key, absolutize)

    # The NAMDgui template funnels every output file through `set output`.
    out_prefix = str(run_dir / spec.name)
    if get_directive(lines, "set output") is not None:
        set_directive(lines, "set output", out_prefix)
    for key in OUTPUT_PATH_KEYS:
        map_directive(
            lines, key,
            lambda v: None if "$" in v else str(run_dir / Path(v).name),
        )
    if get_directive(lines, "set output") is None:
        set_directive(lines, "outputname", out_prefix)

    set_directive(lines, "set temperature", f"{spec.temperature:g}")

    for key, value in overrides.items():
        set_directive(lines, key, value)

    # Tcl action lines at the tail of the file.
    if spec.minimize_steps > 0:
        set_directive(lines, "minimize", spec.minimize_steps)
    else:
        comment_out(lines, "minimize", "skipped: minimize_steps = 0")
    if spec.run_steps > 0:
        set_directive(lines, "run", spec.run_steps)
    else:
        comment_out(lines, "run", "skipped: run_steps = 0")

    header = [
        "# ---------------------------------------------------------------",
        f"# Generated by run_namd_sweep.py on {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"#   run label        {spec.name}",
        f"#   temperature      {spec.temperature:g} K",
        f"#   minimize steps   {spec.minimize_steps}",
        f"#   MD steps         {spec.run_steps}",
        "# ---------------------------------------------------------------",
        "",
    ]
    return "\n".join(header + lines) + "\n"


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

WALL_RE = re.compile(r"WallClock:\s*([\d.eE+-]+).*?CPUTime:\s*([\d.eE+-]+)")


def parse_namd_output(path: Path) -> dict:
    """Pull the energies, timing and any errors out of a NAMD log."""
    cols: list[str] | None = None
    first = last = after_min = None
    wall = cpu = None
    completed = False
    errors: list[str] = []
    n_energy = 0

    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("ETITLE:"):
                cols = line.split()[1:]
            elif line.startswith("ENERGY:"):
                vals = line.split()[1:]
                if cols and len(vals) == len(cols):
                    rec = {}
                    for k, v in zip(cols, vals):
                        try:
                            rec[k] = float(v)
                        except ValueError:
                            rec[k] = v
                    if "TS" in rec:
                        rec["TS"] = int(rec["TS"])
                    if first is None:
                        first = rec
                    last = rec
                    n_energy += 1
            elif line.startswith("TCL: Running for") and after_min is None:
                after_min = last
            elif line.startswith("WallClock:"):
                m = WALL_RE.search(line)
                if m:
                    wall, cpu = float(m.group(1)), float(m.group(2))
            elif "End of program" in line:
                completed = True
            else:
                s = line.strip()
                if s.startswith(("FATAL ERROR", "ERROR:")) and len(errors) < 5:
                    errors.append(s[:300])

    return {
        "first": first or {},
        "last": last or {},
        "after_min": after_min or {},
        "namd_wallclock_s": wall,
        "namd_cputime_s": cpu,
        "namd_completed": completed,
        "n_energy_records": n_energy,
        "errors": errors,
    }


SUMMARY_FIELDS = [
    "label", "temperature_K", "minimize_steps", "md_steps",
    "status", "returncode", "driver_wall_s", "namd_wallclock_s", "namd_cputime_s",
    "final_ts", "final_total", "final_potential", "final_kinetic",
    "final_temp", "final_tempavg",
    "potential_after_min", "total_after_min", "potential_initial",
    "n_energy_records", "conf", "log", "dcd", "coor", "error",
]


def summary_row(spec: SimSpec, run_dir: Path, status: dict) -> dict:
    parsed = status.get("parsed", {})
    last = parsed.get("last", {})
    amin = parsed.get("after_min", {})
    first = parsed.get("first", {})
    pick = lambda d, k: d.get(k) if isinstance(d, dict) else None
    exists = lambda p: str(p) if Path(p).exists() else ""
    return {
        "label": spec.name,
        "temperature_K": f"{spec.temperature:g}",
        "minimize_steps": spec.minimize_steps,
        "md_steps": spec.run_steps,
        "status": status.get("status", "unknown"),
        "returncode": status.get("returncode", ""),
        "driver_wall_s": round(status["driver_wall_s"], 2) if status.get("driver_wall_s") else "",
        "namd_wallclock_s": parsed.get("namd_wallclock_s") or "",
        "namd_cputime_s": parsed.get("namd_cputime_s") or "",
        "final_ts": pick(last, "TS") or "",
        "final_total": pick(last, "TOTAL") or "",
        "final_potential": pick(last, "POTENTIAL") or "",
        "final_kinetic": pick(last, "KINETIC") or "",
        "final_temp": pick(last, "TEMP") or "",
        "final_tempavg": pick(last, "TEMPAVG") or "",
        "potential_after_min": pick(amin, "POTENTIAL") or "",
        "total_after_min": pick(amin, "TOTAL") or "",
        "potential_initial": pick(first, "POTENTIAL") or "",
        "n_energy_records": parsed.get("n_energy_records", ""),
        "conf": exists(run_dir / f"{spec.name}.namd"),
        "log": exists(run_dir / "namd.out"),
        "dcd": exists(run_dir / f"{spec.name}.dcd"),
        "coor": exists(run_dir / f"{spec.name}.coor"),
        "error": "; ".join(parsed.get("errors", []))[:500],
    }


def write_summary(outdir: Path, rows: list[dict]) -> Path:
    csv_path = outdir / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (outdir / "summary.json").write_text(json.dumps(rows, indent=2))
    return csv_path


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

_procs: set[subprocess.Popen] = set()
_procs_lock = threading.Lock()
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def run_one(spec: SimSpec, run_dir: Path, namd_bin: str, threads: int,
            namd_args: list[str]) -> dict:
    conf = run_dir / f"{spec.name}.namd"
    logfile = run_dir / "namd.out"
    cmd = [namd_bin, f"+p{threads}", *namd_args, conf.name]

    log(f"  -> {spec.name}: {' '.join(cmd)}")
    t0 = time.time()
    with open(logfile, "w") as fh:
        fh.write(f"# command: {' '.join(cmd)}\n# cwd: {run_dir}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=run_dir, stdout=fh,
                                stderr=subprocess.STDOUT)
        with _procs_lock:
            _procs.add(proc)
        try:
            rc = proc.wait()
        finally:
            with _procs_lock:
                _procs.discard(proc)
    wall = time.time() - t0

    parsed = parse_namd_output(logfile)
    ok = rc == 0 and parsed["namd_completed"] and not parsed["errors"]
    status = {
        "label": spec.name,
        "temperature": spec.temperature,
        "minimize_steps": spec.minimize_steps,
        "run_steps": spec.run_steps,
        "command": cmd,
        "returncode": rc,
        "driver_wall_s": wall,
        "status": "ok" if ok else "failed",
        "parsed": parsed,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))

    mark = "ok" if ok else "FAILED"
    log(f"  <- {spec.name}: {mark} in {wall:.1f}s "
        f"(final TOTAL={parsed['last'].get('TOTAL', 'n/a')}, "
        f"TEMP={parsed['last'].get('TEMP', 'n/a')})")
    return status


def kill_all() -> None:
    with _procs_lock:
        procs = list(_procs)
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--template", default="initial_run/2jof_streight_autopsf.namd",
                    help="NAMD config to use as the template (default: %(default)s)")
    ap.add_argument("--outdir", default="results",
                    help="directory to hold one subdirectory per run (default: %(default)s)")
    ap.add_argument("-T", "--temperatures", nargs="+", default=["298"],
                    help="temperatures in K; numbers, a,b,c lists or start:stop:step")
    ap.add_argument("-m", "--minimize", nargs="+", default=["1000"],
                    help="minimization step counts (0 disables minimization)")
    ap.add_argument("-d", "--md", nargs="+", default=["10000"],
                    help="MD step counts (0 disables the run command)")
    ap.add_argument("--mode", choices=("product", "zip"), default="product",
                    help="combine the three lists as a cartesian product or "
                         "element-wise (default: %(default)s)")
    ap.add_argument("--sweep", type=Path,
                    help="JSON file describing the sweep; overrides -T/-m/-d")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override any other NAMD directive, e.g. --set dcdfreq=100 "
                         "(repeatable)")
    ap.add_argument("--namd", default="namd2", help="NAMD binary (default: %(default)s)")
    ap.add_argument("--namd-args", default="",
                    help="extra arguments passed to NAMD, e.g. '+setcpuaffinity'")
    ap.add_argument("-p", "--threads", type=int, default=4,
                    help="NAMD threads per job, becomes +p<N> (default: %(default)s)")
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="how many NAMD jobs to run concurrently (default: %(default)s)")
    ap.add_argument("--max-runs", type=int, default=64,
                    help="refuse to start more runs than this (default: %(default)s)")
    ap.add_argument("--estimate", action="store_true",
                    help="print the run matrix and exit without running NAMD")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the config files but do not launch NAMD")
    ap.add_argument("--resume", action="store_true",
                    help="skip runs whose status.json already says ok")
    ap.add_argument("--summarize-only", action="store_true",
                    help="rebuild summary.csv from existing run directories")
    args = ap.parse_args(argv)

    template = Path(args.template).resolve()
    if not template.is_file():
        ap.error(f"template not found: {template}")
    base_dir = template.parent
    # Relative to the current working directory (not the template's
    # directory) so --outdir behaves the way a shell user expects.
    outdir = Path(args.outdir).resolve()

    if args.sweep:
        specs, mode = specs_from_file(args.sweep)
    else:
        specs = build_specs(
            parse_numlist(args.temperatures),
            parse_numlist(args.minimize),
            parse_numlist(args.md),
            args.mode,
        )
        mode = args.mode

    if not specs:
        ap.error("the sweep is empty")

    overrides = {}
    for item in args.sets:
        if "=" not in item:
            ap.error(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()

    total_steps = sum(s.minimize_steps + s.run_steps for s in specs)
    log(f"template      {template}")
    log(f"results dir   {outdir}")
    log(f"sweep mode    {mode}")
    log(f"runs          {len(specs)}")
    log(f"total steps   {total_steps:,}")
    if overrides:
        log(f"overrides     {overrides}")

    if args.estimate:
        log("")
        log(f"{'label':<34}{'T (K)':>8}{'minimize':>10}{'MD':>10}{'steps':>12}")
        for s in specs:
            log(f"{s.name:<34}{s.temperature:>8g}{s.minimize_steps:>10}"
                f"{s.run_steps:>10}{s.minimize_steps + s.run_steps:>12,}")
        log("")
        log("Rerun without --estimate to execute.")
        return 0

    if len(specs) > args.max_runs:
        log(f"\nRefusing to start {len(specs)} runs (--max-runs is {args.max_runs}).")
        log("Shrink the lists, use --mode zip, or raise --max-runs.")
        return 2

    # -- summarize-only: just re-read what is on disk ------------------------
    if args.summarize_only:
        rows = []
        for spec in specs:
            run_dir = outdir / spec.name
            sj = run_dir / "status.json"
            if sj.is_file():
                status = json.loads(sj.read_text())
            elif (run_dir / "namd.out").is_file():
                status = {"status": "unknown", "parsed": parse_namd_output(run_dir / "namd.out")}
            else:
                continue
            rows.append(summary_row(spec, run_dir, status))
        if not rows:
            log("No finished runs found under the results directory.")
            return 1
        log(f"\nsummary -> {write_summary(outdir, rows)}  ({len(rows)} runs)")
        return 0

    namd_bin = shutil.which(args.namd) or args.namd
    if not args.dry_run and not shutil.which(args.namd):
        ap.error(f"NAMD binary not found on PATH: {args.namd}")

    ncpu = os.cpu_count() or 1
    if args.jobs * args.threads > ncpu:
        log(f"warning: {args.jobs} jobs x {args.threads} threads = "
            f"{args.jobs * args.threads} > {ncpu} cores; runs will contend")

    template_lines = template.read_text().splitlines()
    namd_args = args.namd_args.split()
    outdir.mkdir(parents=True, exist_ok=True)

    # -- generate configs ---------------------------------------------------
    pending: list[SimSpec] = []
    for spec in specs:
        run_dir = outdir / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        conf = run_dir / f"{spec.name}.namd"
        conf.write_text(render_config(template_lines, spec, run_dir, base_dir, overrides))
        sj = run_dir / "status.json"
        if args.resume and sj.is_file():
            try:
                if json.loads(sj.read_text()).get("status") == "ok":
                    log(f"  == {spec.name}: already ok, skipping")
                    continue
            except json.JSONDecodeError:
                pass
        pending.append(spec)

    if args.dry_run:
        log(f"\nWrote {len(specs)} config files under {outdir} (dry run, nothing executed).")
        return 0

    # -- run ----------------------------------------------------------------
    log(f"\nrunning {len(pending)} job(s), {args.jobs} at a time, "
        f"+p{args.threads} each\n")
    t0 = time.time()
    results: dict[str, dict] = {}
    try:
        if args.jobs <= 1:
            for spec in pending:
                results[spec.name] = run_one(spec, outdir / spec.name, namd_bin,
                                             args.threads, namd_args)
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futs = {
                    pool.submit(run_one, spec, outdir / spec.name, namd_bin,
                                args.threads, namd_args): spec
                    for spec in pending
                }
                for fut, spec in futs.items():
                    results[spec.name] = fut.result()
    except KeyboardInterrupt:
        log("\ninterrupted, terminating NAMD processes...")
        kill_all()
        return 130

    # -- collect ------------------------------------------------------------
    rows = []
    for spec in specs:
        run_dir = outdir / spec.name
        status = results.get(spec.name)
        if status is None:
            sj = run_dir / "status.json"
            if sj.is_file():
                status = json.loads(sj.read_text())
            else:
                continue
        rows.append(summary_row(spec, run_dir, status))

    csv_path = write_summary(outdir, rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    log("")
    log(f"done in {time.time() - t0:.1f}s: {ok}/{len(rows)} runs ok")
    log(f"summary -> {csv_path}")
    failed = [r["label"] for r in rows if r["status"] != "ok"]
    if failed:
        log(f"failed: {', '.join(failed)}")
        log(f"inspect {outdir}/<label>/namd.out for details")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
