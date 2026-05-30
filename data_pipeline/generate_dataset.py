"""
data_pipeline/generate_dataset.py

Synthetic thermodynamic Q&A dataset generator.

Three-stage pipeline:
    Stage 1 — Question generation    (Gemini 2.5 Pro)
    Stage 2 — Answer generation      (Gemini 2.5 Pro)
    Stage 3 — Scientific vetting     (Gemini 2.5 Flash)

Only entries passing the vetting stage are written to the output JSONL.

Hardware: CPU only. Runs on MacBook or Colab without GPU.

Usage:
    python data_pipeline/generate_dataset.py \
        --pdf your_textbook.pdf \
        --output data/thermo_dataset.jsonl \
        --api_key YOUR_GEMINI_API_KEY
"""

import json
import os
import time
import argparse
import pathlib
from typing import Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("Run: pip install google-genai")

try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def pdf_to_markdown(pdf_path: str, output_path: str) -> str:
    """Convert a PDF textbook to Markdown for chunking."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"📄 Converting PDF: {pdf_path}")

    if pymupdf4llm:
        md_text = pymupdf4llm.to_markdown(pdf_path)
    else:
        import pdfplumber
        lines = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    lines.append(f"\n\n## Page {i}\n\n{text}")
        md_text = "".join(lines)

    pathlib.Path(output_path).write_bytes(md_text.encode("utf-8"))
    print(f"✅ Saved to {output_path} ({len(md_text):,} chars)")
    return md_text


# ---------------------------------------------------------------------------
# Stage 1: Question generation
# ---------------------------------------------------------------------------

def generate_instruction(client, pro_model: str, chunk: str) -> Optional[str]:
    """Generate a single thermodynamics question from a text chunk."""
    prompt = f"""You are a Professor of Chemical Engineering.
Based on the technical data below, generate ONE thermodynamics question for an engineering student.

CRITICAL CONSTRAINTS:
- The question must be a single standalone sentence or short paragraph.
- DO NOT mention "the excerpt", "the text", or "the provided data".
- Mix question types: approximately half conceptual, half quantitative calculation.
- Return ONLY a JSON object with a single key: "instruction".
- The value of "instruction" must be a plain string — no nested keys, no headers.

TECHNICAL DATA: {chunk}"""

    try:
        response = client.models.generate_content(
            model=pro_model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text).get("instruction")
    except Exception as e:
        print(f"  ⚠️  Stage 1 error: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 2: Answer generation
# ---------------------------------------------------------------------------

def generate_answer(client, pro_model: str, instruction: str, chunk: str) -> Optional[dict]:
    """Generate a rigorous step-by-step answer for a thermodynamics question."""
    prompt = f"""You are a Thermodynamics Specialist. Answer the question below rigorously.

QUESTION: {instruction}
CONTEXT: {chunk}

Guidelines:
- MATH: Use LaTeX for all formulas (e.g., $PV^n = C$).
- JSON SAFETY: Double-escape all backslashes (e.g., \\\\Delta, \\\\frac{{P_1}}{{P_2}}).
- REASONING: Provide clear step-by-step derivation or explanation.
- FORMAT: Return a JSON object with exactly two keys:
    "instruction": copy ONLY the question text — no system prompt text, no "QUESTION:" prefix
    "output": your full step-by-step answer as a plain string.
- Do NOT nest additional keys inside either field."""

    try:
        response = client.models.generate_content(
            model=pro_model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  ⚠️  Stage 2 error: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 3: Scientific vetting
# ---------------------------------------------------------------------------

def vet_entry(client, flash_model: str, pair: dict, chunk: str) -> bool:
    """Audit a Q&A pair for scientific accuracy using Gemini Flash."""
    prompt = f"""You are a Technical Auditor. Review this data for scientific accuracy.

SOURCE TEXT: {chunk}
PROPOSED DATA: {json.dumps(pair)}

Criteria:
1. Is the "output" scientifically sound based on the source?
2. Does the "instruction" avoid mentioning the source text?
3. Are backslashes correctly double-escaped?

Respond ONLY with 'VALID' or 'INVALID'."""

    try:
        response = client.models.generate_content(
            model=flash_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return "VALID" in response.text.upper()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    markdown_path: str,
    output_path: str,
    api_key: str,
    chunk_size: int = 2800,
    sleep_seconds: float = 2.0,
    pro_model: str = "gemini-2.5-pro",
    flash_model: str = "gemini-2.5-flash",
) -> None:
    """Run the full 3-stage data generation pipeline."""
    if not os.path.exists(markdown_path):
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    with open(markdown_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    client = genai.Client(api_key=api_key)
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    print(f"📚 {len(full_text):,} characters → {len(chunks)} chunks")
    print(f"🚀 Pipeline: Pro={pro_model} | Flash={flash_model}\n")

    saved = rejected = 0

    for i, chunk in enumerate(chunks):
        print(f"📊 Chunk {i + 1}/{len(chunks)}", end=" | ")

        instruction = generate_instruction(client, pro_model, chunk)
        if not instruction:
            print("❌ Stage 1 failed")
            continue

        # Catch prompt contamination
        if "You are a Thermodynamics" in instruction or "QUESTION:" in instruction:
            print("⚠️  Contaminated instruction — skipping")
            continue

        pair = generate_answer(client, pro_model, instruction, chunk)
        if not pair:
            print("❌ Stage 2 failed")
            continue

        if vet_entry(client, flash_model, pair, chunk):
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(pair) + "\n")
            saved += 1
            print(f"✅ Saved ({saved} total)")
        else:
            rejected += 1
            print(f"🗑️  Rejected ({rejected} total)")

        time.sleep(sleep_seconds)

    print(f"\n{'='*50}")
    print(f"✅ Complete | Saved: {saved} | Rejected: {rejected}")
    print(f"   Accept rate: {saved / max(1, saved + rejected) * 100:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic thermodynamics dataset.")
    parser.add_argument("--pdf", type=str, help="Source PDF path")
    parser.add_argument("--markdown", type=str, default="thermo.md")
    parser.add_argument("--output", type=str, default="data/thermo_dataset.jsonl")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=2800)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Provide --api_key or set GEMINI_API_KEY")

    if args.pdf and not os.path.exists(args.markdown):
        pdf_to_markdown(args.pdf, args.markdown)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    run_pipeline(args.markdown, args.output, api_key, args.chunk_size, args.sleep)
