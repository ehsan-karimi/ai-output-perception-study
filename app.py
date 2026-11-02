"""A user study on how people perceive the attributes in AI-generated images and texts.

    streamlit run app.py

Each participant sees 15 items, one per attribute, in random order; N_TEXT of them show 10 texts, the others 10
images, each from a randomly drawn model (m1, m2, m3). For an item the participant sees the 10 outputs generated from
one sentence and answers (1) which option appears most often, with "No bias" (intensity 0) and "Cannot tell" (the
attribute is not shown or mentioned), and (2) how strongly the outputs lean towards it, from 0 to 10 (eleven
buttons, none preselected, so no default value can anchor the answer).

Answers are appended to responses.jsonl next to this file, one JSON object per line. The secret "institution"
(optional, .streamlit/secrets.toml) names the research institution on the consent page.
"""
from __future__ import annotations

import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
N_ITEMS, N_TEXT = 15, 5
NO_BIAS, CANNOT = "No bias (no option dominates)", "Cannot tell (the attribute is not shown / not mentioned)"
FIELDS = ["timestamp_utc", "participant", "position", "item_id", "bias", "model", "modality", "caption_id",
          "choice", "intensity", "seconds", "app_version"]
APP_VERSION = "2026-11-02"


@st.cache_data
def load_items() -> dict:
    items = json.loads((HERE / "data/items.json").read_text())
    return {i["id"]: i for i in items}


def assign(items: dict, seed: int) -> list[str]:
    """One item per bias; N_TEXT biases in text, the rest in images; a random model per item; random order."""
    rng = random.Random(seed)
    biases = sorted({i["bias"] for i in items.values()})
    text_biases = set(rng.sample(biases, N_TEXT))
    models = sorted({i["model"] for i in items.values()})
    chosen = [f"{b}|{rng.choice(models)}|{'text' if b in text_biases else 'image'}" for b in biases]
    rng.shuffle(chosen)
    return chosen[:N_ITEMS]


def secret(name: str, default=None):
    try:                                          # no secrets.toml at all raises instead of returning the default
        return st.secrets.get(name, default)
    except Exception:
        return default


def save(row: dict) -> None:
    """One JSON object per line in responses.jsonl, next to app.py (append-only, flushed at once). The file only
    survives on a machine whose disk persists: run the app on your own computer or server, not on Streamlit Cloud."""
    with open(HERE / "responses.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def consent() -> None:
    st.title("How do AI models depict and describe the world?")
    st.markdown(
        f"This study is part of academic research{' at ' + secret('institution') if secret('institution') else ''} "
        "on how AI models that generate both images and texts depict people, objects and scenes.\n\n"
        f"You will see **{N_ITEMS} tasks**. In each task you see **10 images or 10 "
        "texts that an AI model generated from the same sentence**, and you answer two questions about them.\n\n"
        "- Participation is voluntary and anonymous: we store only your answers, a random ID and timestamps.\n"
        "- You can stop at any time by closing the page.\n"
        "- The answers are used only for scientific research.")
    if st.checkbox("I am at least 18 years old and I agree to take part in this study."):
        if st.button("Start", type="primary"):
            st.session_state.stage = "intro"
            st.rerun()


def intro() -> None:
    st.header("How to answer")
    st.markdown(
        "Each task names one **attribute** (for example *gender*, *grass color* or *lighting*) and shows 10 outputs "
        "generated from the same sentence.\n\n"
        "1. **Which option appears most often** across the 10 outputs? Choose *No bias* if no option clearly "
        "dominates, and *Cannot tell* if the outputs do not show (images) or do not mention (texts) the attribute.\n"
        "2. **How strongly** do the 10 outputs lean towards that option, from **0** (spread evenly, no bias) to "
        "**10** (all 10 the same).\n\n"
        "There are no right or wrong answers: we want your own judgement.")
    if st.button("Begin", type="primary"):
        st.session_state.stage, st.session_state.t0 = "items", time.time()
        st.rerun()


def show_item(item: dict, pos: int) -> None:
    st.progress(pos / N_ITEMS, text=f"Task {pos + 1} of {N_ITEMS}")
    noun = "images" if item["modality"] == "image" else "texts"
    st.markdown(f"These 10 {noun} were generated from the sentence:  \n> *{item['caption'].strip()}*")
    subject = f" of {item['refer_to']}" if item["refer_to"] else ""
    st.subheader(f"Attribute: {item['bias']}{subject}")
    if item["modality"] == "image":
        for row in range(2):
            cols = st.columns(5)
            for k in range(5):
                cols[k].image(str(HERE / "data" / item["outputs"][row * 5 + k]), width="stretch")
    else:
        for k, t in enumerate(item["outputs"], 1):
            with st.container(border=True):
                st.markdown(f"**Text {k}**  \n{t}")
    # not clear_on_submit: a rejected submit (a question unanswered) must keep the other answer; every task has its
    # own form and widget keys, so the next task starts empty anyway
    with st.form(f"form_{pos}"):
        choice = st.radio(f"1. Which option appears most often across the 10 {noun}?",
                          [c.capitalize() for c in item["classes"]] + [NO_BIAS, CANNOT], index=None,
                          key=f"choice_{pos}")
        intensity = st.radio(f"2. How strongly do the 10 {noun} lean towards that option? "
                             "(0 = evenly spread / no bias, 10 = all the same; skip it if you chose No bias or "
                             "Cannot tell)", list(range(11)), index=None, horizontal=True,
                             key=f"intensity_{pos}")
        if st.form_submit_button("Next", type="primary"):
            if choice is None:
                st.warning("Please choose an option for question 1.")
                return
            if intensity is None and choice not in (NO_BIAS, CANNOT):
                st.warning("Please choose a value for question 2.")
                return
            save({"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "participant": st.session_state.pid, "position": pos + 1, "item_id": item["id"],
                  "bias": item["bias"], "model": item["model"], "modality": item["modality"],
                  "caption_id": item["caption_id"], "choice": choice.lower() if choice not in (NO_BIAS, CANNOT)
                  else ("no bias" if choice == NO_BIAS else "cannot tell"),
                  "intensity": "" if choice == CANNOT else (0 if choice == NO_BIAS else intensity),
                  "seconds": round(time.time() - st.session_state.t0, 1), "app_version": APP_VERSION})
            st.session_state.pos, st.session_state.t0 = pos + 1, time.time()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="AI outputs study", layout="wide")
    if "pid" not in st.session_state:
        st.session_state.pid = uuid.uuid4().hex[:12]
        st.session_state.order = assign(load_items(), seed=int(st.session_state.pid, 16))
        st.session_state.stage, st.session_state.pos = "consent", 0
    stage = st.session_state.stage
    if stage == "consent":
        consent()
    elif stage == "intro":
        intro()
    elif st.session_state.pos < N_ITEMS:
        show_item(load_items()[st.session_state.order[st.session_state.pos]], st.session_state.pos)
    else:
        st.title("Thank you!")
        st.markdown("Your answers have been saved. You can close this page.")


main()
