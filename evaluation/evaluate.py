"""
evaluation/evaluate.py

Gemini-as-judge evaluation framework.

Compares base vs fine-tuned model on thermodynamics questions.
Supports both merged models and adapter-only loading (for large models).

Scoring rubric (max 9 points):
    correctness:  0-4  (is the core answer correct?)
    reasoning:    0-3  (is the step-by-step working sound?)
    clarity:      0-2  (is the answer clearly explained?)
    hallucination: 0/-1 (deduct 1 for confidently wrong facts)

Usage:
    # Merged model (e.g. Llama 3B)
    python evaluation/evaluate.py \
        --base_model meta-llama/Llama-3.2-3B-Instruct \
        --finetuned_model ./outputs/llama-merged \
        --hf_token YOUR_TOKEN \
        --gemini_key YOUR_GEMINI_KEY

    # Adapter-only (e.g. Mistral 7B — avoids 14GB merge)
    python evaluation/evaluate.py \
        --base_model mistralai/Mistral-7B-Instruct-v0.2 \
        --adapter_path ./outputs/mistral-adapter \
        --hf_token YOUR_TOKEN \
        --gemini_key YOUR_GEMINI_KEY
"""

import os
import json
import argparse

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Evaluation questions — drawn from training data topic distribution
# High-frequency topics: enthalpy, efficiency, heat engine
# Low-frequency topics: COP, free expansion, throttling
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = [
    # High-frequency topics
    {
        "id": "Q01", "topic": "Heat Engine Net Work", "frequency": "high",
        "question": "A heat engine operates between a furnace at 900 K and a cooling reservoir at 300 K. In each cycle it absorbs 600 kJ from the furnace. Calculate the maximum work output and the heat rejected.",
        "reference": "Carnot efficiency = 1 - T_L/T_H = 1 - 300/900 = 0.667. Max work = 0.667 * 600 = 400 kJ. Heat rejected = 600 - 400 = 200 kJ."
    },
    {
        "id": "Q02", "topic": "Thermal Efficiency Comparison", "frequency": "high",
        "question": "Two heat engines operate between reservoirs at 800 K and 300 K. Engine A has efficiency 40%, Engine B has efficiency 65%. Which violates the second law and why?",
        "reference": "Carnot efficiency = 1 - 300/800 = 62.5%. Engine A (40%) is below Carnot — irreversible but possible. Engine B (65%) exceeds Carnot — impossible, violates second law."
    },
    {
        "id": "Q03", "topic": "Enthalpy Change", "frequency": "high",
        "question": "Air is heated at constant pressure from 300 K to 600 K. Using cp = 1.005 kJ/(kg·K), calculate the specific enthalpy change and explain why enthalpy is the appropriate property.",
        "reference": "Δh = cp * ΔT = 1.005 * 300 = 301.5 kJ/kg. At constant pressure Q = ΔH, so enthalpy directly equals heat transfer."
    },
    # Low-frequency topics
    {
        "id": "Q04", "topic": "Refrigerator COP", "frequency": "low",
        "question": "A refrigerator has a COP of 2.5. If it consumes 2 kJ of work per cycle, calculate the heat removed from the cold space and the heat rejected to the surroundings.",
        "reference": "COP_R = Q_L / W. Q_L = 2.5 * 2 = 5 kJ removed. Q_H = Q_L + W = 5 + 2 = 7 kJ rejected."
    },
    {
        "id": "Q05", "topic": "Free Expansion", "frequency": "low",
        "question": "A rigid insulated vessel is divided equally. One side has nitrogen at 400 kPa and 350 K, the other is evacuated. The partition ruptures. Find the final pressure and temperature.",
        "reference": "Q=0, W=0, so ΔU=0. For ideal gas T2 = T1 = 350 K. Volume doubles: P2 = P1/2 = 200 kPa."
    },
    {
        "id": "Q06", "topic": "Throttling Process", "frequency": "low",
        "question": "Refrigerant passes through a throttling valve from 1200 kPa to 200 kPa. Explain what happens to enthalpy and temperature, and why throttling is used in refrigeration cycles.",
        "reference": "Throttling is isenthalpic: h1 = h2. For a real refrigerant temperature drops because saturation temperature decreases with pressure, producing cold low-pressure refrigerant for the evaporator."
    },
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model(model_path: str, hf_token: str, adapter_path: str = None):
    """
    Load model for inference.

    If adapter_path is provided, loads base model + LoRA adapter.
    Otherwise loads the model_path directly (merged model).
    """
    is_local = os.path.isdir(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=None if is_local else hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=get_bnb_config(),
        device_map="auto",
        token=None if is_local else hf_token,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(model, tokenizer, question: str, max_new_tokens: int = 400) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a senior Thermodynamics Engineer. Provide clear, accurate answers with step-by-step reasoning.",
        },
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Gemini judge
# ---------------------------------------------------------------------------

def gemini_judge(question: str, reference: str, response: str, client) -> dict:
    prompt = f"""You are an expert thermodynamics professor evaluating a student's answer.

QUESTION: {question}
REFERENCE ANSWER: {reference}
STUDENT RESPONSE: {response}

Return ONLY a JSON object with these exact keys:
{{
  "correctness": <0-4, is the core answer correct?>,
  "reasoning": <0-3, is the step-by-step reasoning sound?>,
  "clarity": <0-2, is the answer clearly explained?>,
  "hallucination": <0 or -1, deduct 1 for confidently stated incorrect facts>,
  "total": <sum of above, minimum 0>,
  "comment": "<one sentence explaining the main strength or weakness>"
}}

Be strict. Correct answer with wrong working scores low on reasoning."""

    try:
        r = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        scores = json.loads(r.text)
        scores["total"] = max(
            0,
            sum(scores.get(k, 0) for k in ["correctness", "reasoning", "clarity", "hallucination"]),
        )
        return scores
    except Exception as e:
        print(f"  ⚠️  Judge error: {e}")
        return {"correctness": 0, "reasoning": 0, "clarity": 0,
                "hallucination": 0, "total": 0, "comment": "Judge failed"}


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(args):
    client = genai.Client(api_key=args.gemini_key)
    results = []

    models_to_eval = [
        ("finetuned", args.finetuned_model or args.base_model, args.adapter_path),
        ("base", args.base_model, None),
    ]

    for label, path, adapter in models_to_eval:
        print(f"\n{'='*55}\n  Evaluating: {label.upper()}\n{'='*55}")
        model, tokenizer = load_model(path, args.hf_token, adapter_path=adapter)

        for q in EVAL_QUESTIONS:
            print(f"  ▶ {q['id']} — {q['topic']} [{q['frequency']} frequency]")
            response = generate(model, tokenizer, q["question"])
            scores = gemini_judge(q["question"], q["reference"], response, client)
            print(f"     Score: {scores['total']}/9 | {scores['comment']}")

            results.append({
                "model": label,
                "id": q["id"],
                "topic": q["topic"],
                "frequency": q["frequency"],
                "response": response,
                "correctness": scores.get("correctness", 0),
                "reasoning": scores.get("reasoning", 0),
                "clarity": scores.get("clarity", 0),
                "hallucination": scores.get("hallucination", 0),
                "total": scores["total"],
                "comment": scores.get("comment", ""),
            })

        del model, tokenizer
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    print("\n" + "=" * 55)
    print("OVERALL RESULTS")
    print("=" * 55)
    print(df.groupby("model")["total"].agg(["mean", "std", "min", "max"]).round(2).to_string())

    print("\nBy topic frequency:")
    print(df.groupby(["model", "frequency"])["total"].mean().round(2).to_string())

    print("\nHallucination count:")
    print(df.groupby("model")["hallucination"].apply(lambda x: (x < 0).sum()).to_string())

    os.makedirs(args.results_dir, exist_ok=True)
    df.to_csv(f"{args.results_dir}/evaluation_results.csv", index=False)
    print(f"\n✅ Saved to {args.results_dir}/evaluation_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--finetuned_model", type=str, default=None,
                        help="Path to merged fine-tuned model (large models use --adapter_path instead)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter folder (loads on top of base_model)")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--gemini_key", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    args.hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    args.gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")

    if not args.hf_token:
        raise ValueError("Provide --hf_token or set HF_TOKEN")
    if not args.gemini_key:
        raise ValueError("Provide --gemini_key or set GEMINI_API_KEY")

    run_evaluation(args)
