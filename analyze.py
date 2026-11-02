"""Compare the participants' answers with a reference answer key, per modality and per model.

    python analyze.py --responses responses.jsonl --key private/answer_key.json [--exclude-participants id1 id2]

--responses: responses.jsonl (local run), or the folder of the downloaded Hugging Face dataset, or a CSV.
--key: per item the reference majority class and intensities (not distributed with the study).

Per item the human intensity is the mean of the 0-10 answers / 10 ("No bias" counts 0, "Cannot tell" is left out).
Reported: the mean absolute error between the human intensity and the reference intensity of the item's attribute
and of the shown sentence; the share of answers naming the reference majority class ("No bias" counts as a different
answer, "Cannot tell" is left out); the share of "Cannot tell" answers; the Spearman correlation between human
intensity and the sentence's reference intensity.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def spearman(x: list[float], y: list[float]) -> float | None:
    def rank(v):
        o = sorted(range(len(v)), key=v.__getitem__)
        r, i = [0.0] * len(v), 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy) if sx and sy else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--items", default=str(HERE / "data/items.json"))
    ap.add_argument("--key", default=str(HERE / "private/answer_key.json"))
    ap.add_argument("--exclude-participants", nargs="*", default=[])
    a = ap.parse_args()
    key = json.loads(Path(a.key).read_text())
    items = {i["id"]: {**i, **key["items"][i["id"]]} for i in json.loads(Path(a.items).read_text())}
    src = Path(a.responses)
    if src.is_dir():                              # the downloaded Hugging Face dataset: every responses-*.jsonl in it
        files = sorted(src.rglob("*.jsonl"))
        raw = [json.loads(line) for f in files for line in open(f) if line.strip()]
    elif src.suffix == ".jsonl":
        raw = [json.loads(line) for line in open(src) if line.strip()]
    else:
        raw = list(csv.DictReader(open(src)))
    rows = [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in raw
            if r["participant"] not in set(a.exclude_participants)]
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)
    print(f"{len(rows)} answers from {len({r['participant'] for r in rows})} participants on {len(by_item)} of "
          f"{len(items)} items")

    models = sorted({i["model"] for i in items.values()})
    for scope in ["image", "text"] + [f"{mod}/{m}" for mod in ("image", "text") for m in models]:
        mod, _, model = scope.partition("/")
        ids = [i for i in by_item if items[i]["modality"] == mod and (not model or items[i]["model"] == model)]
        ans = [r for i in ids for r in by_item[i]]
        if not ans:
            continue
        cannot = sum(r["choice"] == "cannot tell" for r in ans)
        named = [r for r in ans if r["choice"] != "cannot tell"]
        agree = sum(r["choice"] == items[r["item_id"]]["ref_majority"] for r in named)
        hum, bias_sev, cap_sev = [], [], []
        for i in ids:
            v = [float(r["intensity"]) / 10 for r in by_item[i] if r["intensity"] not in ("", None)]
            if v:
                hum.append(sum(v) / len(v))
                bias_sev.append(items[i]["ref_attribute"])
                cap_sev.append(items[i]["ref_sentence"])
        ame = lambda s: sum(abs(h - x) for h, x in zip(hum, s)) / len(hum) if hum else None
        rho = spearman(hum, cap_sev)
        print(f"{scope:16} answers {len(ans):4}  cannot tell {100 * cannot / len(ans):5.1f}%  "
              f"direction {100 * agree / len(named) if named else float('nan'):5.1f}% (of {len(named)})  "
              f"AME bias {ame(bias_sev) if hum else float('nan'):.3f}  AME caption "
              f"{ame(cap_sev) if hum else float('nan'):.3f}  (items {len(hum)})  "
              f"rho {'n/a' if rho is None else f'{rho:.2f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
