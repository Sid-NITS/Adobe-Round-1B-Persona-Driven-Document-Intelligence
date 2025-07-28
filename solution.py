#!/usr/bin/env python3
"""
Hybrid PDF Section + Sub-section extractor (v2 - HyDE)
 1. Gemma3-1B-IT         : generate Hypothetical Document Embeddings (HyDE)
 2. MiniLM-L6-v2         : embeddings          (bi-encoder)
 3. cross-encoder/MS-Marco: accurate re-ranking
 4. Gemma3-1B-IT         : deep subsection summaries

This version first generates a set of "ideal answers" (hypothetical documents)
based on the query, then finds PDF pages that are most similar to those ideal
answers.
"""

# ── Imports ──────────────────────────────────────────────────────────────
import os, re, json, fitz, logging
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from litert_tools.pipeline import pipeline  # Gemma3 wrapper

# ── CONFIG ───────────────────────────────────────────────────────────────
PDF_FOLDER = "Collection 1/PDFs"
INPUT_JSON_PATH = "Collection 1/challenge1b_input.json"
OUTPUT_JSON_PATH = "Collection 1/challenge1b_output.json"

TOP_K_PAGES = 5  # final pages wanted
CANDIDATES_FOR_RERANK = 25  # how many pages flow to cross-encoder
MIN_PAGE_CHARS = 200
MIN_SENT_CHARS = 40
SNIPPET_SENTENCES = 3

# NEW: Number of hypothetical answers to generate for the HyDE step
NUM_HYPOTHETICAL_ANSWERS = 15

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
GEMMA_MODEL_PATH = "gemma3-1b-it-int4.task"  # local .task file

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)

# ── Load models ──────────────────────────────────────────────────────────
logging.info(f"Loading MiniLM bi-encoder: {MINILM_MODEL}")
bi_encoder = SentenceTransformer(MINILM_MODEL)

logging.info(f"Loading cross-encoder re-ranker: {CROSS_MODEL}")
cross_encoder = CrossEncoder(CROSS_MODEL)

logging.info("Loading Gemma3-1B-IT …")
gemma_runner = pipeline.load(GEMMA_MODEL_PATH, repo_id="litert-community/Gemma3-1B-IT")


# ── Helper functions ─────────────────────────────────────────────────────
def get_pages(pdf_path: str) -> List[Dict]:
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for i in range(doc.page_count):
            txt = doc.load_page(i).get_text("text").strip()
            if len(txt) >= MIN_PAGE_CHARS:
                pages.append({"page_number": i + 1, "text": txt})
    except Exception as e:
        logging.error(f"Could not process {pdf_path}: {e}")
    return pages


def create_query_text(ctx: Dict) -> str:
    return f"Role: {ctx['persona']} | Task: {ctx['job']}"


def generate_title(page_text: str) -> str:
    for line in page_text.splitlines():
        line = line.strip()
        if (
            10 < len(line) < 100
            and not line.isupper()
            and line.endswith((".", "?", "!", ":"))
        ):
            return line[:80]
    return page_text[:80].strip().replace("\n", " ") + "…"


def best_sentences(page_text: str) -> str:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", page_text)
        if len(s.strip()) >= MIN_SENT_CHARS
    ]
    if len(sentences) <= SNIPPET_SENTENCES:
        return " ".join(sentences)

    scored = []
    for s in sentences:
        score = len(s)
        if any(
            k in s.lower()
            for k in ("important", "key", "result", "conclusion", "finding")
        ):
            score *= 1.5
        scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return " ".join(s for _, s in scored[:SNIPPET_SENTENCES])


# ── Gemma helpers ────────────────────────────────────────────────────────
def generate_hypothetical_answers(ctx: Dict, n: int) -> List[str]:
    """
    Use Gemma to generate a list of "ideal" answers or information snippets
    that a person with the given persona/task would be looking for.
    This happens *before* reading any documents.
    """
    prompt = (
        f"You are a {ctx['persona']}. Your goal is to '{ctx['job']}'.\n\n"
        f"Generate a numbered list of {n} distinct, ideal answers or key pieces of information "
        "you would hope to find to accomplish your goal. These should be concise, "
        "hypothetical statements. Do not ask questions. Frame them as answers.\n\n"
        "Ideal Answers:\n"
    )
    logging.info("🤖 Generating hypothetical answers with Gemma...")
    try:
        raw = gemma_runner.generate(prompt).strip()
        # Split on numbered list markers. This is more robust.
        parts = re.split(r"^\s*\d+\.\s*", raw, flags=re.M)
        answers = [p.strip(" •-*‐\n") for p in parts if p.strip()]
        if not answers:
            logging.warning(
                "Gemma did not produce a numbered list of answers, using raw output."
            )
            return [raw]
        logging.info(f"✅ Generated {len(answers)} hypothetical answers.")
        return answers
    except Exception as e:
        logging.error(f"Gemma3 error during hypothetical answer generation: {e}")
        return []


def gemma_subsection(text: str, ctx: Dict) -> str:
    prompt = (
        f"Given the task '{ctx['job']}' and persona '{ctx['persona']}', "
        "summarise the key information in this passage in 2-3 sentences.\n\n"
        f"Passage: '''{text}'''\n\nSummary:"
    )
    try:
        out = gemma_runner.generate(prompt).strip()
        return out.lstrip("Summary:").strip() or text[:200] + "…"
    except Exception as e:
        logging.warning(f"Gemma3 subsection summary error: {e}")
        return text[:200] + "…"


# ── 1️⃣  Fast recall (HyDE) ───────────────────────────────────────────────
def recall_by_hyde(
    pages: List[Dict], hypothetical_answer_embeddings: np.ndarray
) -> List[Dict]:
    """
    Ranks pages based on their similarity to a pre-computed set of
    hypothetical answer embeddings.
    """
    if not pages or hypothetical_answer_embeddings.size == 0:
        return []

    logging.info("Embedding %d pages with MiniLM...", len(pages))
    page_texts = [p["text"][:1024] for p in pages]  # Truncate for efficiency
    page_embeddings = bi_encoder.encode(
        page_texts, batch_size=128, show_progress_bar=True
    )

    logging.info("Calculating similarity between pages and hypothetical answers...")
    # This creates a matrix of (num_pages x num_hypothetical_answers)
    similarity_matrix = cosine_similarity(
        page_embeddings, hypothetical_answer_embeddings
    )

    # For each page, find the score of the *best matching* hypothetical answer
    best_scores = np.max(similarity_matrix, axis=1)

    # Attach scores and titles to each page object
    for page, score in zip(pages, best_scores):
        page["bi_sim"] = float(score)
        page["section_title"] = generate_title(page["text"])

    pages.sort(key=lambda x: -x["bi_sim"], reverse=True)
    return pages[:CANDIDATES_FOR_RERANK]


# ── 2️⃣  Cross-encoder re-ranking (unchanged) ────────────────────────────
def rerank_with_cross_encoder(candidates: List[Dict], ctx: Dict) -> List[Dict]:
    if not candidates:
        return []

    logging.info("🎯 Cross-encoder re-ranking %d candidates...", len(candidates))
    query = create_query_text(ctx)
    # The cross-encoder benefits from seeing a bit more text.
    pairs = [(query, c["text"][:2048]) for c in candidates]

    scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=True)
    for c, s in zip(candidates, scores):
        c["cross_score"] = float(s)

    candidates.sort(key=lambda x: -x["cross_score"], reverse=True)
    return candidates[:TOP_K_PAGES]


# ── Core pipeline ────────────────────────────────────────────────────────
def process_documents(payload: Dict) -> Dict:
    ctx = {
        "persona": payload["persona"]["role"],
        "job": payload["job_to_be_done"]["task"],
    }
    logging.info(f"Persona: {ctx['persona']} | Task: {ctx['job']}")

    # -------- HyDE Step 1: Generate & Embed Hypothetical Answers -----------
    hypothetical_answers = generate_hypothetical_answers(
        ctx, n=NUM_HYPOTHETICAL_ANSWERS
    )
    if not hypothetical_answers:
        logging.critical("Could not generate hypothetical answers. Aborting.")
        return {"error": "Failed to generate hypothetical answers."}

    hyde_embeddings = bi_encoder.encode(
        hypothetical_answers, show_progress_bar=True, batch_size=32
    )

    # -------- Collect and Process all PDF pages ----------------------------
    all_pages: List[Dict] = []
    for doc in payload["documents"]:
        pdf_path = os.path.join(PDF_FOLDER, os.path.basename(doc["filename"]))
        if not os.path.exists(pdf_path):
            logging.warning(f"File not found, skipping: {pdf_path}")
            continue
        pages = get_pages(pdf_path)
        for p in pages:
            p["document"] = os.path.basename(doc["filename"])
        all_pages.extend(pages)
        logging.info(f"📄 {os.path.basename(pdf_path)} – kept {len(pages)} pages")

    if not all_pages:
        logging.critical("No pages were extracted from any PDF. Aborting.")
        return {"error": "Failed to extract any text from the provided documents."}

    # -------- Fast Recall (HyDE) -------------------------------------------
    logging.info("⚡ Fast recall (HyDE) on %d pages…", len(all_pages))
    recalled_candidates = recall_by_hyde(all_pages, hyde_embeddings)
    logging.info(
        "→ %d pages recalled for cross-encoder stage", len(recalled_candidates)
    )

    # -------- Cross-encoder Re-ranking -------------------------------------
    top_pages = rerank_with_cross_encoder(recalled_candidates, ctx)
    logging.info("✅ Final top-%d pages selected", len(top_pages))

    # -------- Assemble Final JSON Output -----------------------------------
    extracted_sections = []
    for rank, p in enumerate(top_pages, 1):
        extracted_sections.append(
            {
                "document": p["document"],
                "section_title": p["section_title"],
                "importance_rank": rank,
                "page_number": p["page_number"],
            }
        )

    logging.info("🤖 Running Gemma3 for final subsection summaries…")
    subsection_analysis = []
    for p in tqdm(top_pages, ncols=90, desc="Gemma-Summaries"):
        # We use `best_sentences` here to feed a concise chunk to the final summary model
        snippet = best_sentences(p["text"])
        refined = gemma_subsection(snippet, ctx)
        subsection_analysis.append(
            {
                "document": p["document"],
                "refined_text": refined,
                "page_number": p["page_number"],
            }
        )

    return {
        "metadata": {
            "input_documents": [
                os.path.basename(d["filename"]) for d in payload["documents"]
            ],
            "persona": ctx["persona"],
            "job_to_be_done": ctx["job"],
            "processing_timestamp": datetime.utcnow().isoformat(),
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis,
    }


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("🚀 Starting hybrid PDF pipeline (v2 - HyDE)...")
    try:
        with open(INPUT_JSON_PATH, "r") as f:
            payload = json.load(f)

        result = process_documents(payload)

        # Only try to create a preview if there's no error
        if "error" not in result:
            preview = json.dumps(result, indent=2)[:2000]
            if len(preview) == 2000:
                preview += "..."
            logging.info("✅ Extraction complete – preview ↓\n%s", preview)
            with open(OUTPUT_JSON_PATH, "w") as f:
                json.dump(result, f, indent=2)
            logging.info(f"📁 Saved to: {OUTPUT_JSON_PATH}")
        else:
            logging.error(f"Pipeline failed: {result['error']}")

    except FileNotFoundError:
        logging.critical(f"Input file not found at: {INPUT_JSON_PATH}")
    except json.JSONDecodeError:
        logging.critical(f"Could not decode JSON from: {INPUT_JSON_PATH}")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
