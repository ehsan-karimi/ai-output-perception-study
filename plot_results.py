"""Figure of the study results: one panel per modality; per attribute, the participants' intensity (0-10 answer / 10)
as mean (SD) on the faulty list (triplets the implicit filter removed) next to the cleaned list (triplets it kept).

    python plot_results.py --responses private/answers --out private/fig_userstudy [--banner "PILOT"]

The panel titles give the share of "No bias" answers per list.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
HUMAN, REF = "#0072B2", "#E69F00"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"], "pdf.fonttype": 42})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="a .jsonl file or the folder of the downloaded dataset")
    ap.add_argument("--key", default=str(HERE / "private/answer_key.json"))
    ap.add_argument("--out", default=str(HERE / "private/fig_userstudy"))
    ap.add_argument("--banner", default="")
    a = ap.parse_args()
    src = Path(a.responses)
    files = sorted(glob.glob(str(src / "**/*.jsonl"), recursive=True)) if src.is_dir() else [str(src)]
    rows = [json.loads(line) for f in files for line in open(f) if line.strip()]
    key = json.loads(Path(a.key).read_text())["items"]

    lists = json.loads(Path(a.key).read_text())["lists"]
    items = {i["id"]: {**i, "list": lists[i["group"]]} for i in json.loads((HERE / "data/items.json").read_text())}
    rows = [r for r in rows if r["item_id"] in items]
    biases = sorted({i["bias"] for i in items.values()})
    col = {"faulty": "#d99a2b", "cleaned": "#3b78b0"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (mod, title) in zip(axes, (("image", "Images"), ("text", "Texts"))):
        shares = []
        for k, lst in enumerate(("faulty", "cleaned")):
            rs = [r for r in rows if items[r["item_id"]]["list"] == lst and r["modality"] == mod]
            mean, sd = [], []
            for b in biases:
                v = [float(r["intensity"]) / 10 for r in rs if r["bias"] == b]
                mean.append(st.mean(v) if v else 0)
                sd.append(st.stdev(v) if len(v) > 1 else 0)
            x = [i + (k - 0.5) * 0.38 for i in range(len(biases))]
            ax.bar(x, mean, 0.36, color=col[lst], label=f"{lst} list", linewidth=0, zorder=2)
            yerr = [[min(d, m) for m, d in zip(mean, sd)], [min(d, 1 - m) for m, d in zip(mean, sd)]]  # kept in 0-1
            ax.errorbar(x, mean, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.5, zorder=3)
            shares.append(f"{lst} {100 * sum(r['choice'] == 'no bias' for r in rs) / max(len(rs), 1):.0f}% "
                          f"(n={len(rs)})")
        ax.set_title(f"{title}: No bias answers, " + ", ".join(shares), fontsize=6.4)
        ax.set_xticks(range(len(biases)))
        ax.set_xticklabels(biases, rotation=45, ha="right", rotation_mode="anchor", fontsize=5.6)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", color="#e5e5e5", lw=0.4, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=6, length=2)
    axes[0].set_ylabel("Bias intensity, mean (SD)", fontsize=6.5)
    axes[0].legend(fontsize=5.8, frameon=False, loc="upper left")
    if a.banner:
        fig.text(0.5, 0.55, a.banner, ha="center", va="center", fontsize=15, color="#C00000", alpha=0.35,
                 rotation=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300)
    print(f"wrote {a.out}.pdf/.png ({len(rows)} answers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
