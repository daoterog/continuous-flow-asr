"""Pull the numbers behind paper/results.yaml out of wandb into paper/data/*.csv.

The CSVs are committed, so building the tables and figures never needs wandb. Run this when
an entry gains runs or a run finishes; the build reads only what it wrote.

    python -m scripts.paper.fetch                 # every wandb-backed entry
    python -m scripts.paper.fetch --only phase4   # one entry
    python -m scripts.paper.fetch --compute       # runtimes of every run in every project

An ablation entry writes `<id>.csv` in the same shape the typed tables use (one row per spec:
group, label, role, wer, result, outcome, plus one column per entry `columns:` metric) and
`<id>_curves.csv` (run, epoch, wer), which the curves figures read.

A spec names one run (`run:`) or several seeds of one configuration (`runs: [...]`). Seeds
are averaged, and the result column lists each one, so the spread is in the table.
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import wandb
import yaml

PAPER = Path("paper")
ABLATION_FIELDS = ("group", "label", "role", "wer", "result", "outcome", "rows", "run")


def load_registry(path: Path = PAPER / "results.yaml") -> dict:
    """The registry, as written."""
    return yaml.safe_load(path.read_text())


def history(api: wandb.Api, entity: str, run: str, metric: str) -> dict[int, float]:
    """{epoch: metric} for every epoch that logged it."""
    r = api.run(f"{entity}/{run}")
    out = {}
    for row in r.scan_history(keys=["epoch", metric]):
        if row.get(metric) is not None and row.get("epoch") is not None:
            out[int(row["epoch"])] = float(row[metric])
    return out


def run_ids(spec: dict) -> list[str]:
    """The runs behind a spec: its seeds, or its one run."""
    return list(spec.get("runs") or ([spec["run"]] if "run" in spec else []))


def at_epoch(api: wandb.Api, entity: str, run: str, metric: str, at: int) -> float | None:
    """`metric` at epoch `at`, or None when the run did not log it there."""
    return history(api, entity, run, metric).get(at)


def fetch_ablation(api: wandb.Api, registry: dict, entry: dict) -> list[dict]:
    """One row per reference and ablation spec, at the entry's epoch; and their curves."""
    entity = registry["entity"]
    metric = entry.get("metric", registry["metric"])
    at = int(entry.get("at_epoch", 99))
    extra = entry.get("columns") or []
    rows, curves = [], []
    for group in entry["groups"]:
        members = [(group["reference"], "reference")] + [
            (a, "ablation") for a in group["ablations"]
        ]
        for spec, role in members:
            ids = run_ids(spec)
            if not ids:
                continue
            per_seed = []
            for i, run in enumerate(ids):
                h = history(api, entity, run, metric)
                if at not in h:
                    print(f"  {run}: no {metric} at epoch {at} (has {sorted(h)[-3:]})")
                else:
                    per_seed.append(h[at])
                if i == 0:
                    curves += [
                        {"run": run, "epoch": e, "wer": f"{w:.3f}"} for e, w in sorted(h.items())
                    ]
            row = {
                "group": group["name"],
                "label": spec["label"],
                "role": role,
                "wer": f"{sum(per_seed) / len(per_seed):.2f}" if per_seed else "",
                "result": " / ".join(f"{w:.2f}" for w in per_seed) if len(per_seed) > 1 else "",
                "outcome": spec.get("outcome", ""),
                "rows": " ".join(str(r) for r in entry.get("experiments_rows", [])),
                "run": ids[0],
            }
            for col in extra:
                vals = [at_epoch(api, entity, run, col["metric"], at) for run in ids]
                vals = [v for v in vals if v is not None]
                row[col["name"]] = f"{sum(vals) / len(vals):.3f}" if vals else ""
            rows.append(row)
            label = f"{group['name'][:12]:12s} {spec['label'][:40]:40s}"
            print(f"  {label} {row['wer']} {row['result']}")
    fields = (*ABLATION_FIELDS, *(c["name"] for c in extra))
    write(PAPER / "data" / f"{entry['id']}.csv", rows, fields)
    write(PAPER / "data" / f"{entry['id']}_curves.csv", curves, ("run", "epoch", "wer"))
    return rows


def fetch_compute(api: wandb.Api, registry: dict) -> None:
    """Every run in every project of the entity: when, how long, where, and how it ended.

    `env` is the machine the run's data config pointed at (`snellius` or `local`), which is
    what separates cluster GPU-hours, the billed ones, from the laptop's.
    """
    rows = []
    for project in api.projects(registry["entity"]):
        for r in api.runs(f"{registry['entity']}/{project.name}"):
            data = (r.config or {}).get("data") or {}
            rows.append(
                {
                    "project": project.name,
                    "run": r.id,
                    "name": r.name,
                    "created_at": r.created_at,
                    "runtime_h": f"{(r.summary.get('_runtime') or 0) / 3600:.3f}",
                    "env": data.get("env", ""),
                    "state": r.state,
                }
            )
        print(f"  {project.name}: {sum(1 for x in rows if x['project'] == project.name)} runs")
    write(
        PAPER / "data" / "compute_runs.csv",
        rows,
        ("project", "run", "name", "created_at", "runtime_h", "env", "state"),
    )


def write(path: Path, rows: list[dict], fields: Sequence[str]) -> None:
    """Rows to CSV, columns in `fields` order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch the requested entries, or all wandb-backed ones."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--only", nargs="+", default=None, help="entry ids")
    p.add_argument("--compute", action="store_true", help="refresh compute_runs.csv")
    args = p.parse_args(argv)

    registry = load_registry()
    api = wandb.Api()
    for entry in registry["entries"]:
        if args.only and entry["id"] not in args.only:
            continue
        has_runs = any(
            run_ids(spec)
            for g in entry.get("groups", [])
            for spec in [g["reference"], *g["ablations"]]
        )
        if entry["kind"] == "ablation" and has_runs:
            print(entry["id"])
            fetch_ablation(api, registry, entry)
    if args.compute:
        print("compute")
        fetch_compute(api, registry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
