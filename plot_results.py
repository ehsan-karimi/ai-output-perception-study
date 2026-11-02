"""Figure of the study results: per attribute, the participants' mean intensity next to the
reference intensity of the outputs they saw, one panel per modality.

    python plot_results.py --responses private/answers --out private/fig_userstudy [--banner "PILOT"]

Per item the human intensity is the mean of its 0-10 answers / 10 ("No bias" = 0, "Cannot tell" left out); per
attribute and modality the bars average the items answered. Each panel title gives the direction agreement, the
mean absolute error and the share of "Cannot tell" answers (analyze.py). --banner writes a warning across the figure.
"""
from __future__ import annotations

import argparse
import glob
import json
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

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, (mod, title) in zip(axes, (("image", "Images"), ("text", "Texts"))):
        rs = [r for r in rows if r["modality"] == mod]
        per_item = defaultdict(list)
        for r in rs:
            if r["intensity"] not in ("", None):
                per_item[r["item_id"]].append(float(r["intensity"]) / 10)
        by_bias = defaultdict(lambda: ([], []))
        for i, v in per_item.items():
            b = i.split("|")[0]
            by_bias[b][0].append(sum(v) / len(v))
            by_bias[b][1].append(key[i]["ref_sentence"])
        biases = sorted(by_bias, key=lambda b: sum(by_bias[b][1]) / len(by_bias[b][1]))
        x = range(len(biases))
        hum = [sum(by_bias[b][0]) / len(by_bias[b][0]) for b in biases]
        ref = [sum(by_bias[b][1]) / len(by_bias[b][1]) for b in biases]
        ax.bar([i - 0.2 for i in x], hum, 0.38, color=HUMAN, label="participants", linewidth=0, zorder=2)
        ax.bar([i + 0.2 for i in x], ref, 0.38, color=REF, label="pipeline", linewidth=0, zorder=2)
        named = [r for r in rs if r["choice"] != "cannot tell"]
        agree = sum(r["choice"] == key[r["item_id"]]["ref_majority"] for r in named)
        ame = sum(abs(h - r) for h, r in zip(hum, ref)) / len(hum) if hum else float("nan")
        cannot = sum(r["choice"] == "cannot tell" for r in rs)
        ax.set_title(f"{title}: direction {100 * agree / max(len(named), 1):.0f}%, AME {ame:.2f}, "
                     f"cannot tell {100 * cannot / max(len(rs), 1):.0f}% (n={len(rs)})", fontsize=6.4)
        ax.set_xticks(list(x))
        ax.set_xticklabels(biases, rotation=45, ha="right", rotation_mode="anchor", fontsize=5.6)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#e5e5e5", lw=0.4, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=6, length=2)
    axes[0].set_ylabel("Bias intensity", fontsize=6.5)
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
