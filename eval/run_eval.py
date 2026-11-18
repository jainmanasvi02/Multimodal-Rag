"""Evaluate the retrieval pipeline (CLIP + FAISS + relevance gate) on the sample PDFs.

Run from the repo root:
    python eval/run_eval.py              # retrieval + gate metrics (no API calls, CPU only)
    python eval/run_eval.py --with-llm   # also check final answers with Gemini (uses quota)

Outputs: eval/results.md (summary table for the README) and eval/results.json (per question).

Metrics
  Hit@1 / Hit@k / MRR   the expected source + page + kind (text chunk or image) is among the
                        top-k retrieved items (rank 1 / any rank / reciprocal rank).
  Kept after gate       the expected item is retrieved AND passes its similarity threshold.
  Blocked (off-topic)   no retrieved item passes its threshold, so the graph answers
                        "not found" without calling the LLM.  "Hard" off-topic questions
                        are about the same domain as the PDFs but not answerable from them,
                        so the gate is expected to let some through (the LLM must refuse).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEXT_GRID = [round(0.40 + 0.025 * i, 3) for i in range(23)]  # 0.40 .. 0.95
IMAGE_GRID = [round(0.10 + 0.01 * i, 2) for i in range(25)]  # 0.10 .. 0.34


# ------------------------------------------------------------------ helpers
def load_questions(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def matches(item, q):
    """Does a retrieved item correspond to the place the answer lives?"""
    return item["source"] == q["source"] and item["page"] == q["page"] and item["type"] == q["kind"]


def passes_gate(item, text_thr, image_thr):
    return item["sim"] >= (text_thr if item["type"] == "text" else image_thr)


def gate_open(items, text_thr, image_thr):
    return any(passes_gate(i, text_thr, image_thr) for i in items)


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pct(x):
    return f"{x * 100:.0f}%"


# ------------------------------------------------------------------ retrieval
def run_retrieval(questions, vector_store, k):
    from src.retrieval import retrieve

    results = []
    for q in questions:
        items, seen = [], {}
        for doc, sim in retrieve(vector_store, q["question"], k):
            kind = doc.metadata["type"]
            seen[kind] = seen.get(kind, 0) + 1
            items.append(
                {
                    "rank": seen[kind],
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"],
                    "type": kind,
                    "sim": round(float(sim), 4),
                    "text": re.sub(r"\s+", " ", doc.page_content) if kind == "text" else "",
                }
            )
        results.append({**q, "retrieved": items})
    return results


def matches(item, q):
    """Right source, page and kind; for text questions the chunk must also contain the answer."""
    if not (item["source"] == q["source"] and item["page"] == q["page"] and item["type"] == q["kind"]):
        return False
    if q["kind"] == "text" and "text" in item:
        text = item["text"].lower()
        return any(k.lower() in text for k in q["answer_contains"])
    return True

# ------------------------------------------------------------------ metrics
def summarize(results, text_thr, image_thr):
    def group(rs):
        n = len(rs)
        if n == 0:
            return {"n": 0}
        hit1 = hitk = kept = opened = 0
        mrr = 0.0
        for r in rs:
            items = r["retrieved"]
            rank = next((i["rank"] for i in items if matches(i, r)), None)
            hit1 += rank == 1
            hitk += rank is not None
            mrr += 1.0 / rank if rank else 0.0
            kept += any(matches(i, r) and passes_gate(i, text_thr, image_thr) for i in items)
            opened += gate_open(items, text_thr, image_thr)
        return {
            "n": n,
            "hit@1": hit1 / n,
            "hit@k": hitk / n,
            "mrr": mrr / n,
            "gate_open": opened / n,
            "kept_hit": kept / n,
        }

    def block(rs):
        n = len(rs)
        if n == 0:
            return {"n": 0}
        return {"n": n, "blocked": sum(not gate_open(r["retrieved"], text_thr, image_thr) for r in rs) / n}

    off = [r for r in results if r["kind"] == "offtopic"]
    return {
        "text": group([r for r in results if r["kind"] == "text"]),
        "image": group([r for r in results if r["kind"] == "image"]),
        "all": group([r for r in results if r["kind"] in ("text", "image")]),
        "offtopic_easy": block([r for r in off if r.get("difficulty") == "easy"]),
        "offtopic_hard": block([r for r in off if r.get("difficulty") == "hard"]),
    }


def sweep(results):
    """Grid-search the two thresholds. Score = mean(kept-after-gate rate on answerable
    questions, blocked rate on easy off-topic questions). Returns the centre of the best plateau."""
    answerable = [r for r in results if r["kind"] in ("text", "image")]
    easy = [r for r in results if r["kind"] == "offtopic" and r.get("difficulty") == "easy"]
    if not answerable or not easy:
        return None
    best_score, best = -1.0, []
    for tt in TEXT_GRID:
        for it in IMAGE_GRID:
            kept = _mean(
                any(matches(i, r) and passes_gate(i, tt, it) for i in r["retrieved"]) for r in answerable
            )
            blocked = _mean(not gate_open(r["retrieved"], tt, it) for r in easy)
            score = (kept + blocked) / 2
            if score > best_score + 1e-9:
                best_score, best = score, [(tt, it, kept, blocked)]
            elif abs(score - best_score) <= 1e-9:
                best.append((tt, it, kept, blocked))
    mtt = statistics.median(b[0] for b in best)
    mit = statistics.median(b[1] for b in best)
    tt, it, kept, blocked = min(
        best, key=lambda b: ((b[0] - mtt) / 0.025) ** 2 + ((b[1] - mit) / 0.01) ** 2
    )
    return {
        "text_min_sim": tt,
        "image_min_sim": it,
        "kept_hit": kept,
        "easy_blocked": blocked,
        "score": best_score,
        "n_equally_good": len(best),
    }


# ------------------------------------------------------------------ optional LLM check
def run_llm_eval(results, vector_store, images, sleep):
    from src.graph import build_graph, get_llm

    graph = build_graph(vector_store, images, get_llm())
    for n, r in enumerate(results, start=1):
        try:
            answer = graph.invoke({"query": r["question"], "history": []})["answer"]
            r["llm_error"] = False
        except Exception as exc:  # timeouts, rate limits...
            answer, r["llm_error"] = f"ERROR: {exc}", True
        low = answer.lower()
        if r["kind"] == "offtopic":
            r["llm_correct"] = (not r["llm_error"]) and "could not find" in low
        else:
            r["llm_correct"] = (not r["llm_error"]) and any(k.lower() in low for k in r["answer_contains"])
        r["llm_answer"] = answer
        print(f"  [{n}/{len(results)}] {r['id']}: {'ok' if r['llm_correct'] else 'WRONG'}")
        time.sleep(sleep)


def summarize_llm(results):
    def acc(rs):
        return {"n": len(rs), "correct": _mean(r["llm_correct"] for r in rs)} if rs else {"n": 0}

    off = [r for r in results if r["kind"] == "offtopic"]
    return {
        "text": acc([r for r in results if r["kind"] == "text"]),
        "image": acc([r for r in results if r["kind"] == "image"]),
        "offtopic_easy": acc([r for r in off if r.get("difficulty") == "easy"]),
        "offtopic_hard": acc([r for r in off if r.get("difficulty") == "hard"]),
        "errors": sum(r.get("llm_error", False) for r in results),
    }


# ------------------------------------------------------------------ report
def render_markdown(summary, llm_summary, cfg, suggestion, results):
    t, i, a = summary["text"], summary["image"], summary["all"]
    easy, hard = summary["offtopic_easy"], summary["offtopic_hard"]

    def row(label, key, fmt=pct):
        cells = [fmt(s[key]) if s.get("n") else "-" for s in (t, i, a)]
        return f"| {label} | " + " | ".join(cells) + " |"

    lines = [
        "# Evaluation results",
        "",
        f"Embedding model: `{cfg['clip']}` (CLIP, text and images) | k = {cfg['k']} | "
        f"thresholds: text >= {cfg['text_thr']}, image >= {cfg['image_thr']}",
        f"Corpus: {cfg['n_pdfs']} PDFs, {cfg['n_text']} text chunks, {cfg['n_images']} images | "
        f"Questions: {t['n']} text, {i['n']} image, {easy['n'] + hard['n']} off-topic",
        "",
        "## Retrieval (answerable questions)",
        "",
        "| Metric | Text questions | Image questions | All |",
        "|---|---|---|---|",
        row("Hit@1", "hit@1"),
        row(f"Hit@{cfg['k']}", "hit@k"),
        row("MRR", "mrr", fmt=lambda x: f"{x:.2f}"),
        row("Expected source kept after relevance gate", "kept_hit"),
        row("Relevance gate lets the question through", "gate_open"),
        "",
        "## Refusal (off-topic questions)",
        "",
        "| Type | Questions | Blocked by gate (no LLM call) |",
        "|---|---|---|",
        f"| Easy (unrelated topics) | {easy['n']} | {pct(easy['blocked']) if easy['n'] else '-'} |",
        f"| Hard (same domain, not in the documents) | {hard['n']} | {pct(hard['blocked']) if hard['n'] else '-'} |",
    ]
    if llm_summary:
        ls = llm_summary
        lines += [
            "",
            "## Answer correctness (with Gemini)",
            "",
            "| Question type | Questions | Correct |",
            "|---|---|---|",
            f"| Text | {ls['text']['n']} | {pct(ls['text']['correct'])} |",
            f"| Image | {ls['image']['n']} | {pct(ls['image']['correct'])} |",
            f"| Off-topic, easy (should refuse) | {ls['offtopic_easy']['n']} | {pct(ls['offtopic_easy']['correct'])} |",
            f"| Off-topic, hard (should refuse) | {ls['offtopic_hard']['n']} | {pct(ls['offtopic_hard']['correct'])} |",
            f"",
            f"Requests that errored (counted as wrong): {ls['errors']}",
        ]
    misses = [r for r in results if r["kind"] in ("text", "image") and not any(matches(x, r) for x in r["retrieved"])]
    if misses:
        lines += ["", "## Retrieval misses", ""]
        for r in misses:
            top = r["retrieved"][0] if r["retrieved"] else None
            got = f"{top['source']} p.{top['page'] + 1} ({top['type']}, sim {top['sim']})" if top else "nothing"
            lines.append(f"- `{r['id']}` {r['question']} -> expected {r['source']} p.{r['page'] + 1} ({r['kind']}); top result: {got}")
    if suggestion:
        lines += [
            "",
            "## Suggested thresholds",
            "",
            f"`TEXT_MIN_SIM={suggestion['text_min_sim']}` and `IMAGE_MIN_SIM={suggestion['image_min_sim']}` "
            f"(on this question set: expected source kept {pct(suggestion['kept_hit'])}, "
            f"easy off-topic blocked {pct(suggestion['easy_blocked'])}; "
            f"{suggestion['n_equally_good']} threshold pairs scored equally well). "
            "Tuned on the same questions it is reported on, so treat it as a starting point.",
        ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main
def main():
    from src import config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", default=str(ROOT / "data" / "samples"))
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "eval"))
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--with-llm", action="store_true", help="also check final answers with Gemini")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between LLM calls (rate limits)")
    args = ap.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.pdf_dir}")

    from src.ingest import ingest_files
    from src.retrieval import build_vector_store

    print(f"Indexing {len(pdfs)} PDFs (first run downloads CLIP)...")
    docs, embs, images = ingest_files([(p.name, p.read_bytes()) for p in pdfs])
    n_img = sum(d.metadata["type"] == "image" for d in docs)
    print(f"Indexed {len(docs) - n_img} text chunks and {n_img} images.")
    vector_store = build_vector_store(docs, embs)

    questions = load_questions(args.questions)
    print(f"Running {len(questions)} questions (k={args.k})...")
    results = run_retrieval(questions, vector_store, args.k)

    text_thr, image_thr = config.TEXT_MIN_SIM, config.IMAGE_MIN_SIM
    summary = summarize(results, text_thr, image_thr)
    suggestion = sweep(results)

    llm_summary = None
    if args.with_llm:
        print("Checking answers with Gemini...")
        run_llm_eval(results, vector_store, images, args.sleep)
        llm_summary = summarize_llm(results)

    cfg = {
        "clip": config.CLIP_MODEL_NAME,
        "k": args.k,
        "text_thr": text_thr,
        "image_thr": image_thr,
        "n_pdfs": len(pdfs),
        "n_text": len(docs) - n_img,
        "n_images": n_img,
    }
    report = render_markdown(summary, llm_summary, cfg, suggestion, results)
    print("\n" + report)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(report, encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(
            {"config": cfg, "summary": summary, "llm_summary": llm_summary, "suggested_thresholds": suggestion, "questions": results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved {out / 'results.md'} and {out / 'results.json'}")


if __name__ == "__main__":
    main()
