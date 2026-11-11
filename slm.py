"""slm.py - helpers for Assignment 2: fine-tune a small LM to route support tickets as JSON.

Task (same as Assignment 1): ticket text -> {"category": ..., "urgency": 0-2, "escalate": true/false}
Test set  = the 80 hand-checked tickets from Assignment 1 (data/test_tickets.csv), never used for training.
Train set = synthetic tickets written by a bigger LLM (Claude), then cleaned and label-verified here.

The API key comes from the OPENROUTER_API_KEY environment variable (OpenRouter, OpenAI-compatible API). Never hardcode it.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Task definition
# --------------------------------------------------------------------------- #
CATEGORIES = {
    "billing": "payments, charges, refunds, invoices, subscriptions, promo codes, cashback",
    "delivery": "courier, order status, late or lost parcels, damaged goods, delivery address or time",
    "technical": "app or website bugs, crashes, errors, performance, feature requests, security bugs",
    "account": "login, password, SMS codes, profile data, account deletion, blocked accounts",
}
URGENCY_TEXT = ("0 = question or small annoyance, no time pressure; 1 = real problem, the customer can wait a day; "
                "2 = money at risk, customer is blocked, or a deadline within hours")
ESCALATE_TEXT = ("true if the customer needs a human agent right away: fraud or an unauthorized charge, a data leak, "
                 "a safety or health risk, abuse by staff, or a threat of legal action / police / prosecutor / regulator; else false")

SYSTEM_PROMPT = (
    "You route customer support tickets of an online marketplace in Kazakhstan. Tickets may be in Russian, Kazakh, "
    "English, or a mix. Reply with ONLY a JSON object with exactly these keys:\n"
    '{"category": "billing|delivery|technical|account", "urgency": 0|1|2, "escalate": true|false}\n'
    "Categories: " + "; ".join(f"{k} ({v})" for k, v in CATEGORIES.items()) + ".\n"
    f"urgency: {URGENCY_TEXT}.\nescalate: {ESCALATE_TEXT}."
)

# (urgency, escalate) combinations used when generating training tickets
COMBOS = [(0, False), (1, False), (2, False), (1, True), (2, True)]
LANGS = {
    "ru": "Russian",
    "kk": "Kazakh (Cyrillic script)",
    "en": "English",
    "mix": "Russian mixed with Kazakh words and phrases inside the same message (code-switching, as people in Kazakhstan write)",
}


def answer_json(category: str, urgency: int, escalate: bool) -> str:
    """Canonical target string: compact JSON, fixed key order."""
    return json.dumps({"category": category, "urgency": int(urgency), "escalate": bool(escalate)},
                      ensure_ascii=False, separators=(", ", ": "))


def route_of(category: str, escalate: bool) -> str:
    return "human" if escalate else category


def load_test(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"urgency": int, "escalate": int})
    df["escalate"] = df["escalate"].astype(bool)
    df["output"] = [answer_json(c, u, e) for c, u, e in zip(df["category"], df["urgency"], df["escalate"])]
    return df


# --------------------------------------------------------------------------- #
# Prompt formatting (Qwen3 chat template, thinking switched off)
# --------------------------------------------------------------------------- #
def build_prompt(tokenizer, ticket: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ticket}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def training_text(tokenizer, ticket: str, answer: str) -> str:
    """Exactly the inference prompt + the answer + end-of-turn, so train and test prompts match."""
    return build_prompt(tokenizer, ticket) + answer + "<|im_end|>"


# --------------------------------------------------------------------------- #
# Output parsing and metrics
# --------------------------------------------------------------------------- #
def _check(obj: Any) -> bool:
    return (isinstance(obj, dict) and set(obj) == {"category", "urgency", "escalate"}
            and obj["category"] in CATEGORIES
            and not isinstance(obj["urgency"], bool) and obj["urgency"] in (0, 1, 2)
            and isinstance(obj["escalate"], bool))


def parse_output(text: str) -> dict:
    """strict = the whole reply is a valid JSON object; lenient = the first {...} block is."""
    res = {"obj": None, "strict": False, "lenient": False}
    try:
        obj = json.loads(text.strip())
        if _check(obj):
            return {"obj": obj, "strict": True, "lenient": True}
    except json.JSONDecodeError:
        pass
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        try:
            obj = json.loads(text[i:j + 1])
            if _check(obj):
                res.update(obj=obj, lenient=True)
        except json.JSONDecodeError:
            pass
    return res


def evaluate(test: pd.DataFrame, outputs: list[str], latencies: list[float] | None = None) -> dict:
    """Accuracy uses the lenient parse; an output that is not valid JSON counts as wrong."""
    parsed = [parse_output(o) for o in outputs]
    n = len(test)
    cat = urg = esc = route = exact = 0
    for (_, g), p in zip(test.iterrows(), parsed):
        o = p["obj"]
        if o is None:
            continue
        c_ok, u_ok, e_ok = o["category"] == g["category"], o["urgency"] == g["urgency"], o["escalate"] == g["escalate"]
        cat += c_ok; urg += u_ok; esc += e_ok
        route += route_of(o["category"], o["escalate"]) == g["route"]
        exact += c_ok and u_ok and e_ok
    out = {
        "n": n,
        "valid_json_pct_strict": float(100 * np.mean([p["strict"] for p in parsed])),
        "valid_json_pct": float(100 * np.mean([p["lenient"] for p in parsed])),
        "route_acc": route / n, "category_acc": cat / n, "urgency_acc": urg / n,
        "escalate_acc": esc / n, "exact_match": exact / n,
    }
    if latencies is not None:
        out["latency_mean_s"] = float(np.mean(latencies))
    return out


def run_model(model, tokenizer, tickets: list[str], max_new_tokens: int = 60) -> tuple[list[str], list[float]]:
    """Greedy decoding, one ticket at a time (batch size 1, so latency is per request). Call
    FastLanguageModel.for_inference(model) first."""
    import torch

    texts, lats = [], []
    for t in tickets:
        inputs = tokenizer(build_prompt(tokenizer, t), return_tensors="pt", add_special_tokens=False).to(model.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        lats.append(time.perf_counter() - t0)
        texts.append(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return texts, lats


def side_by_side(test: pd.DataFrame, base_out: list[str], tuned_out: list[str], k: int = 10, seed: int = 7) -> pd.DataFrame:
    """10 test prompts (2 escalations + 8 others, random but fixed) with gold / base / fine-tuned outputs."""
    rng = np.random.RandomState(seed)
    esc = test.index[test["escalate"]].to_numpy()
    rest = test.index[~test["escalate"]].to_numpy()
    idx = list(rng.choice(esc, 2, replace=False)) + list(rng.choice(rest, k - 2, replace=False))
    cut = lambda s: s if len(s) <= 140 else s[:137] + "..."
    rows = []
    for i in idx:
        g = test.loc[i]
        ok = lambda o: (parse_output(o)["obj"] or {}) == json.loads(g["output"])
        rows.append({"id": g["id"], "ticket": cut(g["text"]), "gold": g["output"],
                     "base": cut(base_out[i]), "base_correct": ok(base_out[i]),
                     "fine_tuned": cut(tuned_out[i]), "fine_tuned_correct": ok(tuned_out[i])})
    return pd.DataFrame(rows)


def plot_loss(log_history: list[dict], path: str, title: str = "Training loss"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    ax.plot(*zip(*pts), color="#2a78d6", lw=1.8)
    ax.set_xlabel("step", color="#52514e"); ax.set_ylabel("loss", color="#52514e")
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(True, color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, facecolor="#fcfcfb"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Dataset: generate with a bigger LLM, then clean
# --------------------------------------------------------------------------- #
def make_client():
    from openai import OpenAI

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def _ask(client, model: str, user: str, system: str | None = None, max_tokens: int = 4000) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    r = client.chat.completions.create(model=model, max_tokens=max_tokens, messages=msgs)
    return r.choices[0].message.content or ""


def _json_array(text: str):
    i, j = text.find("["), text.rfind("]")
    if i == -1 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


def _gen_prompt(cat: str, urg: int, esc: bool, lang: str, n: int) -> str:
    return (
        f"Write {n} DIFFERENT realistic customer support tickets that customers of an online marketplace in Kazakhstan "
        f"(orders, wallet payments, courier delivery, mobile app) might send.\n"
        f"Language: {LANGS[lang]}.\n"
        f"All of them must have: main topic = {cat} ({CATEGORIES[cat]}); urgency = {urg} ({URGENCY_TEXT}); "
        f"needs a human right away = {'YES (' + ESCALATE_TEXT.split(': ', 1)[1] + ')' if esc else 'NO'}.\n"
        "Vary length (from 4 words to 4 sentences), tone (calm, angry, polite, sarcastic), and style. Some should contain typos or "
        "missing punctuation, some order numbers or amounts in tenge. Do not mention the labels. Do not number the tickets.\n"
        f"Return ONLY a JSON array of {n} strings."
    )


def generate_raw(client, model: str, n_per_cell: int = 10, workers: int = 6) -> pd.DataFrame:
    """One Claude call per (category, urgency/escalate combo, language) cell -> n tickets with INTENDED labels."""
    cells = [(c, u, e, l) for c in CATEGORIES for (u, e) in COMBOS for l in LANGS]

    def one(cell):
        c, u, e, l = cell
        for _ in range(2):  # one retry on malformed output
            arr = _json_array(_ask(client, model, _gen_prompt(c, u, e, l, n_per_cell), max_tokens=4000))
            if isinstance(arr, list):
                return [dict(text=s, category=c, urgency=u, escalate=e, lang=l) for s in arr if isinstance(s, str)]
        return []

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for res in ex.map(one, cells) for r in res]
    return pd.DataFrame(rows)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def _shingles(s: str, n: int = 5) -> set:
    s = _norm(s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def _jacc(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def relabel(client, model: str, df: pd.DataFrame, batch: int = 10, workers: int = 6) -> pd.DataFrame:
    """Ask the big LLM to label each ticket blindly (it does not see the intended labels)."""
    system = ("You label customer support tickets. For each ticket return category (one of "
              + ", ".join(f"{k}: {v}" for k, v in CATEGORIES.items()) + f"), urgency ({URGENCY_TEXT}), and escalate ({ESCALATE_TEXT}). "
              'Return ONLY a JSON array of objects {"i": <ticket number>, "category": ..., "urgency": 0|1|2, "escalate": true|false}.')
    chunks = [df.iloc[i:i + batch] for i in range(0, len(df), batch)]

    def one(chunk):
        body = "\n".join(f"{k}. {t}" for k, t in enumerate(chunk["text"]))
        for _ in range(2):
            arr = _json_array(_ask(client, model, body, system=system, max_tokens=1500))
            if isinstance(arr, list):
                got = {a.get("i"): a for a in arr if isinstance(a, dict)}
                return [got.get(k) for k in range(len(chunk))]
        return [None] * len(chunk)

    with ThreadPoolExecutor(workers) as ex:
        labels = [a for res in ex.map(one, chunks) for a in res]
    out = df.copy()
    out["ver_category"] = [a.get("category") if a else None for a in labels]
    out["ver_urgency"] = [a.get("urgency") if a else None for a in labels]
    out["ver_escalate"] = [a.get("escalate") if a else None for a in labels]
    return out


def clean_dataset(raw: pd.DataFrame, test: pd.DataFrame, client=None, verify_model: str | None = None,
                  max_rows: int = 1000, near_dup: float = 0.8, leak: float = 0.6, seed: int = 0):
    """Cleaning pipeline. Returns (train_df, stats). Every step records how many rows it removed."""
    stats = {"raw": len(raw)}
    df = raw.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["text"].str.len() >= 8) & (df["text"].str.len() <= 600)]
    stats["after_length_filter(8-600 chars)"] = len(df)

    df["_k"] = df["text"].map(_norm)
    df = df.drop_duplicates("_k").drop(columns="_k")
    stats["after_exact_dedup"] = len(df)

    test_sh = [_shingles(t) for t in test["text"]]
    keep, kept_sh = [], []
    for t in df["text"]:
        sh = _shingles(t)
        if any(_jacc(sh, o) >= near_dup for o in kept_sh) or any(_jacc(sh, o) >= leak for o in test_sh):
            keep.append(False)
        else:
            keep.append(True); kept_sh.append(sh)
    df = df[keep]
    stats[f"after_near_dup(>={near_dup}) and test_leak(>={leak})"] = len(df)

    if client is not None:
        v = relabel(client, verify_model, df)
        ok = ((v["ver_category"] == v["category"]) & (v["ver_urgency"] == v["urgency"]) & (v["ver_escalate"] == v["escalate"]))
        stats["label_check_disagreed"] = int((~ok).sum())
        df = v[ok].drop(columns=["ver_category", "ver_urgency", "ver_escalate"])
        stats["after_label_verification"] = len(df)

    df = df.sample(frac=1, random_state=seed).head(max_rows).reset_index(drop=True)
    df["instruction"] = df["text"]
    df["output"] = [answer_json(c, u, e) for c, u, e in zip(df["category"], df["urgency"], df["escalate"])]
    stats["final_train_rows"] = len(df)
    stats["by_language"] = df["lang"].value_counts().to_dict()
    stats["by_category"] = df["category"].value_counts().to_dict()
    stats["escalate_share"] = round(float(df["escalate"].mean()), 3)
    return df[["instruction", "output", "lang", "category", "urgency", "escalate"]], stats


def write_jsonl(df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in df.to_dict("records"):
            f.write(json.dumps(r, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
