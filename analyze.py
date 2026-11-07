"""What the participants say on the two lists: faulty (triplets the implicit filter removed) and cleaned (triplets it
kept). No judge is involved.

    python analyze.py --responses responses.jsonl [--key private/answer_key.json] [--exclude-participants id1 id2]

--responses: responses.jsonl (local run), or the folder of the downloaded Hugging Face dataset, or a CSV.
--key: the list behind each group code (not distributed).

Per list, modality and attribute: the number of answers, the share of each special option (No bias, Offlist,
Not_visible) and the intensity (the 0-10 answer / 10) as mean (SD).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECIAL = ("no bias", "offlist", "not_visible")


def describe(ans: list[dict]) -> str:
    v = [float(r["intensity"]) / 10 for r in ans]
    sd = stats.stdev(v) if len(v) > 1 else 0.0
    return (f"answers {len(ans):4}  " + "  ".join(f"{s} {100 * sum(r['choice'] == s for r in ans) / len(ans):5.1f}%"
                                                for s in SPECIAL) + f"  intensity {sum(v) / len(v):.2f} ({sd:.2f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--items", default=str(HERE / "data/items.json"))
    ap.add_argument("--key", default=str(HERE / "private/answer_key.json"))
    ap.add_argument("--exclude-participants", nargs="*", default=[])
    a = ap.parse_args()
    lists = json.loads(Path(a.key).read_text())["lists"]
    items = {i["id"]: {**i, "list": lists[i["group"]]} for i in json.loads(Path(a.items).read_text())}
    src = Path(a.responses)
    if src.is_dir():                              # the downloaded Hugging Face dataset: every responses-*.jsonl in it
        raw = [json.loads(line) for f in sorted(src.rglob("*.jsonl")) for line in open(f) if line.strip()]
    elif src.suffix == ".jsonl":
        raw = [json.loads(line) for line in open(src) if line.strip()]
    else:
        raw = list(csv.DictReader(open(src)))
    rows = [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in raw
            if str(r["participant"]) not in set(a.exclude_participants)]
    stale = [r for r in rows if r["item_id"] not in items]          # answers to an older version of the items
    rows = [r for r in rows if r["item_id"] in items]
    if stale:
        print(f"skipped {len(stale)} answers whose item is not in the current items (app versions "
              f"{sorted({r.get('app_version', '?') for r in stale})})")
    print(f"{len(rows)} answers from {len({r['participant'] for r in rows})} participants on "
          f"{len({r['item_id'] for r in rows})} of {len(items)} items")
    biases = sorted({i["bias"] for i in items.values()})
    for mod in ("all", "image", "text"):
        for lst in ("faulty", "cleaned"):
            ans = [r for r in rows if items[r["item_id"]]["list"] == lst and mod in ("all", r["modality"])]
            if ans:
                print(f"{lst:7} {mod:5} {'all attributes':18} {describe(ans)}")
    for b in biases:
        for lst in ("faulty", "cleaned"):
            ans = [r for r in rows if items[r["item_id"]]["list"] == lst and r["bias"] == b]
            if ans:
                print(f"{lst:7} all   {b:18} {describe(ans)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
