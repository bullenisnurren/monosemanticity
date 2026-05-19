#!/usr/bin/env python3
"""
Build a human-readable HTML report from pre-computed SAE features.

Pipeline position:    train.py -> infer.py -> analyse.py

This script does **no** GPU work and never invokes the SAE or the language
model under study.  It only:

  1. Loads the features memmap produced by infer.py (chunk-wise to bound RAM).
  2. Identifies dead features and randomly samples a configurable number of
     non-dead ones (ANALYSIS_NUM_FEATURES).
  3. For each sampled feature, finds the top-k sequences (where the per-token
     max activation is highest).
  4. Asks an OpenAI-compatible LLM to describe what those top-k examples have
     in common.
  5. Renders an HTML report with metadata, descriptions, and per-token green
     highlighting proportional to the feature's activation on each token.

Output: ANALYSIS_DIR / "report.html"
"""

import html as _html
import json
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

import numpy as np
from tqdm import tqdm

from constants import (
    ACTIVATIONS_TRAIN_DIR,
    ANALYSIS_DIR,
    ANALYSIS_NUM_FEATURES,
    ANALYSIS_SEED,
    ANALYSIS_TOP_K,
    DATASET_NAME,
    FEATURES_DIR,
    LLM_API_BASE_URL,
    LLM_API_KEY,
    LLM_API_MODEL,
    LLM_API_TIMEOUT,
    LLM_EXAMPLE_CHARS,
    LLM_NUM_EXAMPLES,
    MODEL_NAME,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_features_meta() -> dict:
    meta_path = FEATURES_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No features meta.json at {meta_path}. Run infer.py first."
        )
    with open(meta_path) as f:
        return json.load(f)


def _load_sequences(path: Path) -> list[dict]:
    """Read sequences.jsonl lazily into a list of dicts."""
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Top-k discovery (uses pre-computed side arrays + per-feature reads)
# ---------------------------------------------------------------------------

def _find_top_k(features_mm: np.memmap,
                sampled_ids: np.ndarray,
                max_per_seq: np.ndarray,
                argmax_per_seq: np.ndarray,
                top_k: int) -> dict[int, list[dict]]:
    """For each sampled feature, return its ``top_k`` sequences (sorted
    descending by peak activation), each carrying the full per-token
    activation row used for rendering.

    The features memmap is feature-major ``(F, N, S)``, so ``features_mm[fid]``
    is a 2-MB contiguous read.  We use ``max_per_seq`` / ``argmax_per_seq``
    (precomputed in infer.py) to pick the top-k sequences without touching
    the big tensor, then fetch the per-token rows for only those sequences.
    """
    out: dict[int, list[dict]] = {}
    for fid in tqdm(sampled_ids, desc="top-k"):
        fid_int = int(fid)
        col = max_per_seq[:, fid_int].astype(np.float32)         # (N,)
        # Indices of sequences with non-zero peak, sorted descending.
        nonzero = np.where(col > 0)[0]
        if nonzero.size == 0:
            out[fid_int] = []
            continue
        k = min(top_k, nonzero.size)
        # argpartition gives the unsorted top-k; we then sort just those.
        cand = nonzero[np.argpartition(-col[nonzero], k - 1)[:k]]
        cand = cand[np.argsort(-col[cand])]
        # Single contiguous 2-MB read per feature.
        feat_rows = np.asarray(features_mm[fid_int], dtype=np.float32)  # (N, S)
        examples = []
        for seq_idx in cand:
            examples.append({
                "val": float(col[seq_idx]),
                "seq_idx": int(seq_idx),
                "peak_idx": int(argmax_per_seq[seq_idx, fid_int]),
                "row": feat_rows[seq_idx].copy(),
            })
        out[fid_int] = examples
    return out


# ---------------------------------------------------------------------------
# LLM-based feature description
# ---------------------------------------------------------------------------

def _build_prompt(top_examples: list[tuple[list[str], int]]) -> str:
    """Build a description prompt from a feature's top examples.

    *top_examples* is a list of ``(token_strings, peak_token_idx)`` pairs.
    Each example is truncated to a window around the peak token.
    """
    lines = [
        "You are an interpretability researcher.  Below are short text "
        "excerpts that all strongly activate the SAME feature in a sparse "
        "autoencoder trained on a transformer language model.  The token "
        "where the feature fires most strongly is wrapped in <<>>.",
        "",
        "Identify what concept, pattern, or behaviour the feature is "
        "detecting.  Reply with ONE concise sentence (max ~20 words), "
        "no preamble, no quotes.",
        "",
    ]
    for i, (tokens, peak_idx) in enumerate(top_examples, 1):
        excerpt = _excerpt_around_peak(tokens, peak_idx, LLM_EXAMPLE_CHARS)
        lines.append(f"Example {i}: {excerpt}")
    return "\n".join(lines)


def _excerpt_around_peak(tokens: list[str], peak_idx: int, max_chars: int) -> str:
    """Return a string with `<<peak_token>>` highlighted, capped to ``max_chars``."""
    parts = list(tokens)
    if 0 <= peak_idx < len(parts):
        parts[peak_idx] = f"<<{parts[peak_idx]}>>"
    full = "".join(parts)
    if len(full) <= max_chars:
        return full
    # Center window on the peak token.
    pre_chars = sum(len(t) for t in parts[:peak_idx])
    half = max_chars // 2
    start = max(0, pre_chars - half)
    end = min(len(full), pre_chars + half)
    snippet = full[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(full):
        snippet = snippet + "..."
    return snippet


def _call_llm(prompt: str) -> str | None:
    """Call an OpenAI-compatible /v1/chat/completions endpoint via stdlib."""
    url = LLM_API_BASE_URL.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": LLM_API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 80,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=LLM_API_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (URLError, TimeoutError, KeyError, ValueError) as exc:
        print(f"[analyse] LLM call failed: {exc!r}")
        return None


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_HTML_TEMPLATE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAE Feature Report - {model}</title>
<style>
  :root {{ --accent: #2a9d8f; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 960px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid var(--accent); padding-bottom: 0.25em; }}
  h2 {{ margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  table.meta td {{ padding: 2px 14px 2px 0; vertical-align: top; }}
  table.meta td:first-child {{ color: #666; white-space: nowrap; }}
  .feature {{ margin: 2em 0; padding: 1em 1.2em; border: 1px solid #e3e3e3;
              border-radius: 6px; background: #fafafa; }}
  .feature h3 {{ margin: 0 0 0.3em 0; }}
  .feature .desc {{ font-style: italic; color: #444; margin-bottom: 0.7em; }}
  .feature .stats {{ font-size: 0.85em; color: #666; margin-bottom: 0.7em; }}
  .examples {{ display: flex; flex-direction: column; gap: 6px; }}
  .example {{ font-family: ui-monospace, "SF Mono", Menlo, monospace;
              font-size: 12px; line-height: 1.45;
              padding: 6px 8px; background: #fff;
              border: 1px solid #eee; border-radius: 4px;
              white-space: pre-wrap; word-break: break-word; }}
  .example .max-act {{ float: right; color: #999; font-size: 11px; }}
  .tok {{ padding: 0 1px; border-radius: 2px; }}
</style>
</head>
<body>
<h1>SAE Feature Report</h1>
"""

_HTML_TEMPLATE_FOOT = """</body>\n</html>\n"""


def _green_bg(intensity: float) -> str:
    """Return a `background-color:` CSS fragment.  *intensity* in [0, 1]."""
    if intensity <= 0:
        return ""
    intensity = max(0.0, min(1.0, intensity))
    # A lightish-to-medium green.  Alpha encodes intensity.
    return f"background-color: rgba(46, 204, 113, {intensity:.3f});"


def _render_example(tokens: list[str], values: np.ndarray, vmax: float) -> str:
    """Render one example as a sequence of <span class=tok> elements."""
    parts = []
    if vmax <= 0:
        vmax = 1.0
    for tok, v in zip(tokens, values):
        intensity = float(v) / vmax if v > 0 else 0.0
        style = _green_bg(intensity)
        text = _html.escape(tok).replace("\n", "↵\n")
        if style:
            parts.append(f'<span class="tok" style="{style}">{text}</span>')
        else:
            parts.append(f'<span class="tok">{text}</span>')
    return "".join(parts)


def _render_meta_table(rows: list[tuple[str, str]]) -> str:
    out = ['<table class="meta">']
    for k, v in rows:
        out.append(f"<tr><td>{_html.escape(k)}</td>"
                   f"<td>{_html.escape(str(v))}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyse():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load features metadata + memmap ---------------------------------
    feat_meta = _load_features_meta()
    features_path = FEATURES_DIR / feat_meta["features_file"]
    features = np.load(str(features_path), mmap_mode="r")  # (F, N, S)
    layout = feat_meta.get("features_layout", "feature_major")
    if layout != "feature_major":
        raise RuntimeError(
            f"Unsupported features_layout {layout!r}.  Re-run infer.py."
        )
    dict_size, n_seqs, seq_len = features.shape

    sequences_path = FEATURES_DIR / feat_meta["sequences_file"]
    sequences = _load_sequences(sequences_path)
    if len(sequences) != n_seqs:
        print(f"[analyse] WARNING: sequences.jsonl has {len(sequences)} entries "
              f"but features have {n_seqs} sequences.")

    fire_count = np.load(str(FEATURES_DIR / feat_meta["fire_count_file"]))
    max_per_seq = np.load(str(FEATURES_DIR / feat_meta["max_per_seq_file"]),
                           mmap_mode="r")
    argmax_per_seq = np.load(str(FEATURES_DIR / feat_meta["argmax_per_seq_file"]),
                              mmap_mode="r")
    n_dead = int((fire_count == 0).sum())
    n_alive = dict_size - n_dead
    print(f"[analyse] Features: {dict_size}, alive: {n_alive}, dead: {n_dead}")

    # ---- Sample non-dead features ----------------------------------------
    rng = np.random.default_rng(ANALYSIS_SEED)
    alive_ids = np.where(fire_count > 0)[0]
    if alive_ids.size == 0:
        raise RuntimeError("No alive features in the test set — nothing to analyse.")
    n_sample = min(ANALYSIS_NUM_FEATURES, alive_ids.size)
    sampled_ids = np.sort(rng.choice(alive_ids, size=n_sample, replace=False))
    print(f"[analyse] Sampling {n_sample} non-dead features for the report.")

    # ---- Find top-k sequences for each sampled feature -------------------
    top = _find_top_k(features, sampled_ids,
                      max_per_seq, argmax_per_seq,
                      ANALYSIS_TOP_K)

    # ---- LLM descriptions -------------------------------------------------
    descriptions: dict[int, str] = {}
    print(f"[analyse] Requesting LLM descriptions from {LLM_API_BASE_URL} "
          f"(model={LLM_API_MODEL}) ...")
    for fid in tqdm(sampled_ids, desc="describe"):
        fid_int = int(fid)
        examples = top[fid_int][:LLM_NUM_EXAMPLES]
        if not examples:
            descriptions[fid_int] = "(no activations)"
            continue
        ex_inputs = [
            (sequences[ex["seq_idx"]]["tokens"], ex["peak_idx"])
            for ex in examples
        ]
        prompt = _build_prompt(ex_inputs)
        desc = _call_llm(prompt)
        descriptions[fid_int] = desc or "(LLM call failed)"

    # ---- Render report ----------------------------------------------------
    report_path = ANALYSIS_DIR / "report.html"
    html_parts: list[str] = []
    html_parts.append(_HTML_TEMPLATE_HEAD.format(
        model=_html.escape(MODEL_NAME)))

    # ---- Metadata sections ----------------------------------------------
    test_meta = feat_meta.get("test_split_meta", {})
    train_cfg = feat_meta.get("training_config", {})
    train_meta_path = ACTIVATIONS_TRAIN_DIR / "meta.json"
    train_meta = json.loads(train_meta_path.read_text()) if train_meta_path.exists() else {}

    html_parts.append("<h2>Experiment metadata</h2>")
    html_parts.append(_render_meta_table([
        ("Model", MODEL_NAME),
        ("d_model", feat_meta.get("d_model", "?")),
        ("Layer index", test_meta.get("layer_index", "?")),
        ("Dataset", DATASET_NAME),
        ("Train tokens", f"{train_meta.get('num_tokens', '?'):,}"
                          if isinstance(train_meta.get("num_tokens"), int)
                          else str(train_meta.get("num_tokens", "?"))),
        ("Test tokens", f"{test_meta.get('num_tokens', '?'):,}"
                         if isinstance(test_meta.get("num_tokens"), int)
                         else str(test_meta.get("num_tokens", "?"))),
        ("Sequence length", test_meta.get("seq_len", seq_len)),
        ("Activation scale", f"{feat_meta.get('activation_scale', float('nan')):.6f}"),
    ]))

    html_parts.append("<h2>SAE metadata</h2>")
    html_parts.append(_render_meta_table([
        ("Checkpoint", feat_meta.get("checkpoint", "?")),
        ("Step", feat_meta.get("step", "?")),
        ("Expansion factor", feat_meta.get("expansion_factor", "?")),
        ("Dictionary size", dict_size),
        ("Alive features (test)", f"{n_alive:,}"),
        ("Dead features (test)", f"{n_dead:,} ({100 * n_dead / dict_size:.2f}%)"),
        ("Batch size", train_cfg.get("batch_size", "?")),
        ("L1 coefficient", train_cfg.get("l1_coeff", "?")),
        ("Learning rate", train_cfg.get("lr", "?")),
        ("Training steps", train_cfg.get("num_training_steps", "?")),
        ("Decoder init norm", train_cfg.get("decoder_init_norm", "?")),
    ]))

    # ---- Per-feature blocks ----------------------------------------------
    html_parts.append("<h2>Sampled features</h2>")
    html_parts.append("<p>Each feature shows a natural-language description "
                       "(generated by an LLM from the top examples) followed "
                       "by the top-k test sequences in which the feature "
                       "fires most strongly.  The greener a token's "
                       "background, the higher the feature's activation on "
                       "that token (normalised within the feature).</p>")

    for fid in sampled_ids:
        fid_int = int(fid)
        examples = top[fid_int]
        desc = descriptions[fid_int]
        peak = examples[0]["val"] if examples else 0.0
        html_parts.append('<div class="feature">')
        html_parts.append(f'<h3>Feature #{fid_int}</h3>')
        html_parts.append(f'<p class="desc">{_html.escape(desc)}</p>')
        html_parts.append(
            f'<p class="stats">Top examples: {len(examples)}; '
            f'peak activation: {peak:.3f}; '
            f'fires on {int(fire_count[fid_int]):,} test tokens '
            f'({100 * int(fire_count[fid_int]) / (n_seqs * seq_len):.4f}%)</p>'
        )
        html_parts.append('<div class="examples">')

        vmax = peak if peak > 0 else 1.0

        for ex in examples:
            tokens = sequences[ex["seq_idx"]]["tokens"]
            rendered = _render_example(tokens, ex["row"], vmax)
            html_parts.append(
                '<div class="example">'
                f'<span class="max-act">max={ex["val"]:.3f} '
                f'(seq #{ex["seq_idx"]}, tok #{ex["peak_idx"]})</span>'
                f'{rendered}</div>'
            )

        html_parts.append('</div></div>')

    html_parts.append(_HTML_TEMPLATE_FOOT)
    report_path.write_text("".join(html_parts))

    print(f"[analyse] Report saved to {report_path}")
    print(f"[analyse] {n_sample} features described, "
          f"{sum(1 for d in descriptions.values() if d.startswith('('))} failed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    analyse()


if __name__ == "__main__":
    main()
