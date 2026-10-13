r"""Turn paper/results.yaml and paper/data/ into tables, figures and RESULTS.md.

Reads only committed files, so it runs anywhere, without wandb:

    python -m scripts.paper.build              # everything
    python -m scripts.paper.build --rerun      # also re-run the scripts behind `command:` figures

Writes, for each entry:
  paper/tables/<id>.tex       booktabs, for \\input{} in Overleaf
  paper/figures/<id>.png      300 dpi
and once:
  paper/RESULTS.md            the whole story in phase order, for reading on GitHub
  paper/all_results.tex       every table and figure in order, for \\input{} while drafting

A `planned` or `running` entry becomes a placeholder in all three, so the paper's skeleton
exists before its numbers do.
"""

import argparse
import csv
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

from scripts.paper.style import INK_SECONDARY, REFERENCE, SERIES, plt, save

PAPER = Path("paper")
SBU_PER_GPU_HOUR = 128  # jobs/common/budget.sh
OUTCOME = {
    "adopt": "adopted",
    "reject": "rejected",
    "neutral": "no effect",
    "kept": "kept",
    # Re-tested on a later stack: shown, but not counted again on the ladder.
    "confirmed": "confirmed",
    # Promising, not settled: the next experiment, not part of the recipe.
    "open": "open lead",
    # Helped, but a variant of it helped more and was adopted instead.
    "superseded": "superseded",
}


# -- data -------------------------------------------------------------------------------


def read_csv(name: str) -> list[dict]:
    """paper/data/<name>.csv as dicts, or [] when it does not exist yet."""
    path = PAPER / "data" / f"{name}.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def number(x: str | None) -> float | None:
    """A CSV cell as a float, or None when empty."""
    return float(x) if x not in (None, "") else None


def ablation_rows(entry: dict) -> list[dict]:
    """The entry's rows with `delta` (vs its group's reference) filled in.

    Placeholder rows come from the registry for entries whose runs have not landed.
    """
    rows = read_csv(entry["id"])
    if not rows and entry.get("groups"):
        rows = [
            {
                "group": g["name"],
                "label": spec["label"],
                "role": role,
                "wer": "",
                "result": "",
                "outcome": spec.get("outcome", ""),
                "rows": "",
            }
            for g in entry["groups"]
            for spec, role in [(g["reference"], "reference")]
            + [(a, "ablation") for a in g["ablations"]]
        ]
    refs = {r["group"]: number(r["wer"]) for r in rows if r["role"] == "reference"}
    for r in rows:
        wer, ref = number(r["wer"]), refs.get(r["group"])
        r["delta"] = None if r["role"] == "reference" or wer is None or ref is None else wer - ref
    return rows


def value(r: dict) -> str:
    """What a row shows in its result column.

    The WER, with the per-seed values beside it when there were several; a free-text result
    (a loss, a share of the batch) when the row has no WER at all.
    """
    if not r.get("wer"):
        return r.get("result") or "--"
    mean = f"{float(r['wer']):.2f}"
    return f"{mean} ({r['result']})" if r.get("result") else mean


# -- tables -----------------------------------------------------------------------------


def value_tex(r: dict) -> str:
    """`value` for LaTeX, where per-seed lists are too wide: the mean +- half the range."""
    try:
        seeds = [float(x) for x in (r.get("result") or "").split(" / ")]
    except ValueError:  # a free-text result, not a list of seeds
        seeds = []
    if r.get("wer") and len(seeds) > 1:
        return rf"{float(r['wer']):.2f}\,$\pm$\,{(max(seeds) - min(seeds)) / 2:.2f}"
    return tex_escape(value(r))


def tex_escape(s: str) -> str:
    """Escape the characters LaTeX treats as markup."""
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("~", r"$\sim$"),
    ):
        s = s.replace(a, b)
    return s.replace("->", r"$\to$")


def extra_columns(entry: dict) -> list[dict]:
    """The entry's `columns:` beyond WER, each {name, metric, header?, digits?}."""
    return entry.get("columns") or []


def cell(r: dict, col: dict) -> str:
    """One extra column's value, rounded as the column asks."""
    v = number(r.get(col["name"]))
    return "" if v is None else f"{v:.{int(col.get('digits', 2))}f}"


def ablation_tex(entry: dict, rows: list[dict], phase: dict) -> str:
    """A booktabs table: one block per group, its reference first, then the changes."""
    extra = extra_columns(entry)
    width = 4 + len(extra)
    # Numbers right-aligned in a fixed width that wraps a long header, so a table with
    # several diagnostic columns still fits the text width; the change column takes the rest.
    num = r">{\raggedleft\arraybackslash}p{" + ("1.0cm" if extra else "1.6cm") + "}"
    # A row listing its seeds ("12.11 (11.68 / 12.54)") needs room to stay on one line.
    seeded = any(r["wer"] and r["result"] for r in rows)
    wer_col = r">{\raggedleft\arraybackslash}p{1.8cm}" if seeded else num
    head = " & ".join(
        ["Change", "WER", r"$\Delta$"]
        + [tex_escape(c.get("header", c["name"])) for c in extra]
        + ["Outcome"]
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize" if len(extra) > 2 else r"\small",
        r"\setlength{\tabcolsep}{3pt}" if len(extra) > 2 else "",
        rf"\begin{{tabularx}}{{\linewidth}}{{@{{}}X{wer_col}{num * (1 + len(extra))}l@{{}}}}",
        r"\toprule",
        head + r" \\",
        r"\midrule",
    ]
    groups = list(dict.fromkeys(r["group"] for r in rows))
    for i, group in enumerate(groups):
        members = [r for r in rows if r["group"] == group]
        best = min((float(r["wer"]) for r in members if r["wer"]), default=None)
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{width}}}{{@{{}}l}}{{\textit{{{tex_escape(group)}}}}} \\")
        for r in members:
            shown = value_tex(r)
            if r["wer"] and best is not None and float(r["wer"]) == best:
                shown = rf"\textbf{{{shown}}}"
            delta = "" if r["delta"] is None else f"{r['delta']:+.2f}"
            label = tex_escape(r["label"]) + (" (reference)" if r["role"] == "reference" else "")
            outcome = OUTCOME.get(r["outcome"], "")
            extras = "".join(f" & {cell(r, c)}" for c in extra)
            lines.append(rf"\quad {label} & {shown} & {delta}{extras} & {outcome} \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    caption = tex_escape(entry["question"])
    setup = tex_escape(entry.get("setup") or phase.get("setup", ""))
    if setup:
        caption += rf" \emph{{Setup: {setup}.}}"
    if any(r"$\pm$" in value_tex(r) for r in rows):
        caption += r" Several seeds: mean $\pm$ half their range."
    if entry["status"] != "done":
        caption += rf" \emph{{({entry['status']}: results pending.)}}"
    lines += [rf"\caption{{{caption}}}", rf"\label{{tab:{entry['id']}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def ablation_md(rows: list[dict], entry: dict) -> str:
    """The same table as Markdown, for RESULTS.md."""
    extra = extra_columns(entry)
    heads = ["change", "WER", "Δ", *(c.get("header", c["name"]) for c in extra), "outcome"]
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for group in dict.fromkeys(r["group"] for r in rows):
        out.append(f"| **{group}** |" + " |" * (len(heads) - 1))
        for r in (r for r in rows if r["group"] == group):
            label = r["label"] + (" *(reference)*" if r["role"] == "reference" else "")
            delta = "" if r["delta"] is None else f"{r['delta']:+.2f}"
            cells = [label, value(r), delta, *(cell(r, c) for c in extra)]
            out.append("| " + " | ".join(cells) + f" | {OUTCOME.get(r['outcome'], '')} |")
    return "\n".join(out)


def plain_tex(entry: dict, rows: list[dict]) -> str:
    """A typed table rendered column for column."""
    cols = list(rows[0])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabularx}}{{\linewidth}}{{@{{}}{'X' * len(cols)}@{{}}}}",
        r"\toprule",
        " & ".join(c.capitalize() for c in cols) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(tex_escape(r[c]) for c in cols) + r" \\" for r in rows]
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        rf"\caption{{{tex_escape(entry['question'])}}}",
        rf"\label{{tab:{entry['id']}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def plain_md(rows: list[dict]) -> str:
    """A typed table as Markdown."""
    cols = list(rows[0])
    return "\n".join(
        ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        + ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    )


def placeholder_tex(entry: dict) -> str:
    """A table-shaped stub, so the paper compiles around results that do not exist yet."""
    return (
        "\\begin{table}[t]\n\\centering\n\\fbox{\\parbox{0.8\\linewidth}{\\centering "
        f"\\textit{{{entry['status']}: {tex_escape(entry['question'])}}}}}}}\n"
        f"\\caption{{{tex_escape(entry['question'])}}}\n\\label{{tab:{entry['id']}}}\n"
        "\\end{table}\n"
    )


# -- figures ----------------------------------------------------------------------------


def first_run(spec: dict) -> str:
    """The run a spec's curve is drawn from: its first seed."""
    return (spec.get("runs") or [spec.get("run")])[0]


def curves_figure(entry: dict, registry: dict) -> Path | None:
    """Sampled WER against epoch, one panel per group of the ablation it plots.

    The reference is the neutral dashed line; each change keeps one categorical slot across
    panels, assigned in registry order, so a colour always means the same change.
    """
    source = next(e for e in registry["entries"] if e["id"] == entry["of"])
    curves = defaultdict(dict)
    for r in read_csv(f"{source['id']}_curves"):
        curves[r["run"]][int(r["epoch"])] = float(r["wer"])
    if not curves:
        return None
    groups = source["groups"]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 2.6), sharey=True)
    slot = 0
    for ax, group in zip(np.atleast_1d(axes), groups, strict=True):
        ref = curves[first_run(group["reference"])]
        ax.plot(
            list(ref),
            list(ref.values()),
            "--",
            color=REFERENCE,
            lw=1.4,
            label=group["reference"]["label"],
        )
        for spec in group["ablations"]:
            h = curves[first_run(spec)]
            color = SERIES[slot % len(SERIES)]
            slot += 1
            ax.plot(list(h), list(h.values()), "-", color=color, label=spec["label"])
            last = max(h)
            ax.plot([last], [h[last]], "o", color=color, ms=4, mec="white", mew=1)
        ax.set_title(group["name"])
        ax.set_xlabel("epoch")
        ax.set_ylim(*entry.get("ylim", (12, 45)))
        ax.legend(loc="lower left", fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("sampled WER (%)")
    return save(fig, PAPER / "figures" / f"{entry['id']}.png")


def audio_info_figure(entry: dict) -> Path | None:
    """Recovery by input and t (top), against where the schedule draws its batch (bottom).

    The draw is the simulated curve the probed run trained under, at its settled mu accuracy,
    rebuilt exactly. Two panels rather than two y-axes on one.
    """
    from src.flow.coefficients import make_schedule
    from src.flow.schedule import TimeReparametrization

    rows = read_csv("audio_info")
    if not rows:
        return None
    t = np.array([float(r["t"]) for r in rows])
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.2, 4.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    series = (
        ("denoiser", "denoiser (audio + mu + x_t)", SERIES[0]),
        ("no_mu", "denoiser with mu removed", SERIES[1]),
        ("field", "x_t alone (argmax)", REFERENCE),
    )
    for key, label, color in series:
        val = [float(r[f"{key}_val"]) for r in rows]
        train = [float(r[f"{key}_train"]) for r in rows]
        top.plot(t, val, "-o", color=color, ms=3, label=f"{label}, val")
        top.plot(t, train, ":", color=color, lw=1.2, label=f"{label}, train")
    top.axhline(0.416, color=SERIES[2], lw=1, ls="--")
    top.text(0.0, 0.43, "mu argmax, val", color=INK_SECONDARY, fontsize=6)
    top.set_ylabel("token recovery")
    top.legend(fontsize=6, loc="lower right")

    curve = TimeReparametrization.exact(
        1193,
        make_schedule("linear_down"),
        make_schedule("linear_up"),
        make_schedule("linear_down"),
        mu_accuracy=0.4776,
    )
    grid = np.linspace(0, 1, 801)
    density = np.gradient(np.asarray(curve.u_of(grid)), grid)
    bottom.fill_between(grid, density, color=REFERENCE, alpha=0.35, lw=0)
    bottom.plot(grid, density, color=REFERENCE, lw=1.2)
    low = float(curve.u_of(0.5))
    bottom.text(
        0.25,
        density.max() * 0.6,
        f"{low:.0%} of the batch\nbelow t = 0.5",
        ha="center",
        fontsize=7,
        color=INK_SECONDARY,
    )
    bottom.set_ylabel("draw density")
    bottom.set_xlabel("flow time t")
    return save(fig, PAPER / "figures" / f"{entry['id']}.png")


def ladder_rows(registry: dict) -> list[dict]:
    """Every adopted change, in phase order, with its gain where it has a number."""
    out = []
    for entry in registry["entries"]:
        if entry["kind"] != "ablation" or entry["status"] != "done":
            continue
        for r in ablation_rows(entry):
            if r["outcome"] == "adopt":
                # A typed row may name itself for the ladder, where it is read without its
                # group ("every 5 epochs" means nothing out of the "tau refresh" block).
                label = r.get("ladder") or r["label"]
                out.append(
                    {
                        "phase": entry["phase"],
                        "group": r["group"],
                        "label": label,
                        "delta": r["delta"],
                        "result": value(r),
                    }
                )
    return out


def ladder_figure(entry: dict, registry: dict) -> Path | None:
    """WER points gained by each adopted change, one bar each, coloured by phase.

    Gains are measured inside their own phase, against that phase's reference, so the bars
    say how much each change bought where it was tested — the phases' absolute WERs are on
    different setups and are not compared.
    """
    rows = [r for r in ladder_rows(registry) if r["delta"] is not None]
    if not rows:
        return None
    phases = sorted({r["phase"] for r in rows})
    titles = {p["id"]: p["title"] for p in registry["phases"]}
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows, strict=True):
        color = SERIES[phases.index(r["phase"]) % len(SERIES)]
        gain = -r["delta"]
        ax.barh(yi, gain, color=color, height=0.62)
        ax.text(gain + 0.3, yi, f"{gain:.1f}", va="center", fontsize=6.5, color=INK_SECONDARY)
    ax.set_yticks(y, [f"{r['label']}" for r in rows], fontsize=6.5)
    ax.set_xlabel("WER points gained, within its phase")
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES[i % len(SERIES)]) for i in range(len(phases))
    ]
    ax.legend(handles, [f"{p}. {titles[p]}" for p in phases], fontsize=6, loc="lower right")
    return save(fig, PAPER / "figures" / f"{entry['id']}.png")


def sbu_range(gpu_hours: float) -> str:
    """The SBU a run's hours cost: alone in a 2-GPU job, or sharing it with a second run."""
    return f"{gpu_hours * SBU_PER_GPU_HOUR:,.0f}-{2 * gpu_hours * SBU_PER_GPU_HOUR:,.0f}"


def compute_summary(entry: dict, registry: dict) -> tuple[list[dict], list[dict]]:
    """(per-phase table rows, per-run rows) from compute_runs.csv.

    Cluster GPU-hours are run runtimes on `snellius`. A run sat alone or with a partner in a
    job billed for at least 2 GPUs, so its SBU lies between 1x and 2x its own hours; both
    ends are reported. Local runs cost no SBU and are counted in hours only.
    """
    runs = read_csv("compute_runs")
    overrides = entry.get("runtime_overrides_h") or {}
    phase_of = entry.get("project_phase") or {}
    for r in runs:
        h = float(overrides.get(r["run"], r["runtime_h"]))
        r["hours"] = h if 0 <= h < 500 else 0.0
        r["phase"] = phase_of.get(r["project"], "other")
        r["cluster"] = r["env"] == "snellius"
    titles = {p["id"]: p["title"] for p in registry["phases"]}
    table = []
    for phase in sorted({r["phase"] for r in runs}, key=lambda p: (isinstance(p, str), p)):
        mine = [r for r in runs if r["phase"] == phase]
        cluster = sum(r["hours"] for r in mine if r["cluster"])
        table.append(
            {
                "phase": f"{phase}. {titles.get(phase, phase)}" if phase != "other" else "other",
                "runs": str(len(mine)),
                "cluster GPU-h": f"{cluster:.1f}",
                "local GPU-h": f"{sum(r['hours'] for r in mine if not r['cluster']):.1f}",
                "est. SBU": sbu_range(cluster),
            }
        )
    cluster = sum(r["hours"] for r in runs if r["cluster"])
    table.append(
        {
            "phase": "total (training runs)",
            "runs": str(len(runs)),
            "cluster GPU-h": f"{cluster:.1f}",
            "local GPU-h": f"{sum(r['hours'] for r in runs if not r['cluster']):.1f}",
            "est. SBU": sbu_range(cluster),
        }
    )
    budget = registry.get("budget", {})
    if budget.get("spent_sbu"):
        table.append(
            {
                "phase": f"accounted (Snellius, as of {budget.get('as_of', '?')})",
                "runs": "",
                "cluster GPU-h": "",
                "local GPU-h": "",
                "est. SBU": f"{budget['spent_sbu']:,} of {budget.get('cap_sbu', 0):,}",
            }
        )
    return table, runs


def compute_figure(entry: dict, registry: dict) -> Path | None:
    """Cumulative GPU-hours over the project, cluster and local, by when each run ended."""
    _, runs = compute_summary(entry, registry)
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    for cluster, label, color in (
        (True, "Snellius A100", SERIES[0]),
        (False, "local GPU", SERIES[1]),
    ):
        mine = sorted(
            (
                datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                + timedelta(hours=r["hours"]),
                r["hours"],
            )
            for r in runs
            if r["cluster"] == cluster
        )
        if not mine:
            continue
        ends, hours = zip(*mine, strict=True)
        total = np.cumsum(hours)
        ax.step(ends, total, where="post", color=color, label=label)
        ax.text(
            ends[-1], total[-1], f" {total[-1]:.0f} h", va="center", fontsize=7, color=INK_SECONDARY
        )
    ax.set_ylabel("GPU-hours, cumulative")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    return save(fig, PAPER / "figures" / f"{entry['id']}.png")


def command_figure(entry: dict, rerun: bool) -> Path | None:
    """Run an existing script's figure (when missing or asked to) and copy it in."""
    produced = Path(entry["produces"])
    if rerun or not produced.exists():
        print(f"  running: {entry['command']}")
        subprocess.run(shlex.split(entry["command"]), check=True)
    if not produced.exists():
        return None
    dest = PAPER / "figures" / f"{entry['id']}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, dest)
    return dest


def field_recovery_figure(entry: dict) -> Path:
    """The field's own token recovery against t, at the project's vocabularies and mu accuracies.

    What a uniform draw over t would spend its batch on.
    """
    from src.flow.coefficients import make_schedule
    from src.flow.recovery import recovery_curve

    t = np.linspace(0, 1, 401)
    down, up = make_schedule("linear_down"), make_schedule("linear_up")
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    cases = (
        (1193, 0.0, "V = 1,193 (tuning), mu accuracy 0"),
        (1193, 0.5, "V = 1,193, mu accuracy 0.5"),
        (51866, 0.0, "V = 51,866 (final), mu accuracy 0"),
    )
    for (vocab, acc, label), color in zip(cases, SERIES, strict=False):
        ax.plot(t, recovery_curve(t, vocab, down, up, down, acc), color=color, label=label)
    ax.axvspan(0, 0.6, color=REFERENCE, alpha=0.12, lw=0)
    ax.text(
        0.3,
        0.55,
        "field carries almost\nnothing: t < 0.6",
        ha="center",
        fontsize=7,
        color=INK_SECONDARY,
    )
    ax.set_xlabel("flow time t")
    ax.set_ylabel("tokens recoverable from x_t")
    ax.legend(loc="upper left", fontsize=6)
    return save(fig, PAPER / "figures" / f"{entry['id']}.png")


def mu_reliance_figure(entry: dict) -> Path | None:
    """Recovery at t = 0 under three versions of mu, one row per arm.

    At t = 0 the field holds no data, so what the denoiser recovers it reads off the audio
    and mu. Giving it its own mu, another utterance's mu, and no mu separates the two: the
    wider the spread, the more the arm's answer is mu's rather than the audio's.
    """
    rows = {}
    for source in entry["sources"]:
        for r in read_csv(source):
            rows.setdefault(r["label"], r)
    arms = [(a["label"], a.get("show", a["label"])) for a in entry["arms"]]
    arms = [(key, shown) for key, shown in arms if key in rows]
    if not arms:
        return None
    readings = (
        ("rec_mu", "its own mu", SERIES[0], "o"),
        ("rec_wrong_mu", "another utterance's mu", SERIES[3], "D"),
        ("rec_no_mu", "no mu (uniform)", SERIES[1], "s"),
    )
    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(arms) + 0.9))
    y = np.arange(len(arms))[::-1]
    for yi, (key, _) in zip(y, arms, strict=True):
        vals = [number(rows[key].get(col)) for col, *_ in readings]
        present = [v for v in vals if v is not None]
        ax.plot([min(present), max(present)], [yi, yi], color=REFERENCE, lw=1, zorder=1)
        for (_, label, color, marker), v in zip(readings, vals, strict=True):
            if v is not None:
                ax.plot(
                    v,
                    yi,
                    marker,
                    color=color,
                    ms=5,
                    mec="white",
                    mew=0.8,
                    zorder=2,
                    label=label if yi == y[0] else None,
                )
    ax.set_yticks(y, [shown for _, shown in arms], fontsize=7)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("token recovery at t = 0 (val)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6)
    return save(fig, PAPER / "figures" / f"{entry['id']}.png")


BUILDERS = {
    "audio_info": audio_info_figure,
    "field_recovery": field_recovery_figure,
    "mu_reliance": mu_reliance_figure,
}


# -- assembly ---------------------------------------------------------------------------


def figure_tex(entry: dict) -> str:
    """A figure environment around the entry's PNG."""
    return (
        "\\begin{figure}[t]\n\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{figures/{entry['id']}.png}}\n"
        f"\\caption{{{tex_escape(entry['question'])}}}\n\\label{{fig:{entry['id']}}}\n"
        "\\end{figure}\n"
    )


def note(entry: dict) -> str | None:
    """notes/<id>.md, or the note an entry points at, if written."""
    path = PAPER / "notes" / f"{entry.get('note', entry['id'])}.md"
    return path.read_text().strip() if path.exists() else None


def build_entry(entry: dict, registry: dict, rerun: bool) -> tuple[str, str]:
    """(LaTeX, Markdown) for one entry, writing its table or figure on the way."""
    phase = next(p for p in registry["phases"] if p["id"] == entry["phase"])
    md = [f"### {entry['question']}", f"`{entry['id']}` · {entry['kind']} · **{entry['status']}**"]
    text = note(entry)
    if text and entry["kind"] != "note":
        md.append(text)
    tex = ""
    kind, done = entry["kind"], entry["status"] == "done"

    if kind == "note":
        md.append(text or "*(note not written yet)*")
    elif kind == "ablation":
        rows = ablation_rows(entry)
        tex = ablation_tex(entry, rows, phase)
        md.append(ablation_md(rows, entry))
    elif kind == "table":
        rows = read_csv(entry["id"])
        if entry["id"] == "compute":
            rows, _ = compute_summary(entry, registry)
        if rows and done:
            tex, table = plain_tex(entry, rows), plain_md(rows)
        else:
            tex, table = placeholder_tex(entry), f"*{entry['status']}: pending*"
        md.append(table)
    elif kind in ("figure", "curves"):
        path = None
        if done:
            if kind == "curves":
                path = curves_figure(entry, registry)
            elif "command" in entry:
                path = command_figure(entry, rerun)
            elif entry.get("builder") == "ladder":
                path = ladder_figure(entry, registry)
                rows = ladder_rows(registry)
                table = [
                    {
                        "phase": str(r["phase"]),
                        "change": r["label"],
                        "result": r["result"],
                        "gain": "" if r["delta"] is None else f"{-r['delta']:.2f}",
                    }
                    for r in rows
                ]
                md.append(plain_md(table))
                tex = plain_tex(entry, table)
                (PAPER / "tables" / "ladder.tex").write_text(tex)
            elif entry.get("builder") == "compute":
                path = compute_figure(entry, registry)
                table, _ = compute_summary(entry, registry)
                md.append(plain_md(table))
                tex = plain_tex(entry, table)
                (PAPER / "tables" / "compute.tex").write_text(tex)
            else:
                path = BUILDERS[entry["builder"]](entry)
        if path:
            tex += figure_tex(entry)
            md.append(f"![{entry['id']}](figures/{entry['id']}.png)")
        else:
            md.append(f"*figure {entry['status']}*")

    if tex and kind in ("ablation", "table"):
        (PAPER / "tables" / f"{entry['id']}.tex").write_text(tex)
    if entry.get("experiments_rows"):
        md.append(f"Lab notebook: EXPERIMENTS.md rows {entry['experiments_rows']}.")
    return tex, "\n\n".join(md)


def main(argv: Sequence[str] | None = None) -> int:
    """Build every entry, then the two assembled documents."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rerun", action="store_true", help="re-run `command:` figures")
    p.add_argument("--only", nargs="+", default=None)
    args = p.parse_args(argv)

    registry = yaml.safe_load((PAPER / "results.yaml").read_text())
    for d in ("tables", "figures"):
        (PAPER / d).mkdir(parents=True, exist_ok=True)
    md = [
        "# Results",
        "",
        "Generated by `python -m scripts.paper.build` from "
        "`paper/results.yaml`. Everything before phase 6 is the tuning phase: compact "
        "vocabulary (V=1,193), train.100.",
        "",
    ]
    tex = ["% Generated by scripts/paper/build.py. \\input{all_results} while drafting."]
    for phase in registry["phases"]:
        entries = [e for e in registry["entries"] if e["phase"] == phase["id"]]
        if not entries:
            continue
        md.append(f"## {phase['id']}. {phase['title']}")
        if phase.get("setup"):
            md.append(f"*Setup: {phase['setup']}.*")
        tex.append(f"\\subsection*{{{phase['id']}. {tex_escape(phase['title'])}}}")
        for entry in entries:
            if args.only and entry["id"] not in args.only:
                continue
            print(entry["id"])
            t, m = build_entry(entry, registry, args.rerun)
            md.append(m)
            if t:
                tex.append(t)
    (PAPER / "RESULTS.md").write_text("\n\n".join(md) + "\n")
    (PAPER / "all_results.tex").write_text("\n".join(tex) + "\n")
    print("wrote paper/RESULTS.md, paper/all_results.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
