# ThermoLLM: Domain Adaptation of LLMs for Thermodynamics Reasoning

> Investigating whether a synthetic chain-of-thought data pipeline can improve thermodynamics reasoning in small language models — under consumer hardware constraints.

**Author:** Egameh Omokagbo | Process Engineer & ML Practitioner  
**Hardware:** MacBook (data pipeline) + Google Colab GPU (training)  
**Models tested:** Llama-3.2-3B-Instruct | Mistral-7B-Instruct-v0.2

---

## Overview

This project asks a focused research question:

> *Can a multi-stage LLM pipeline generate chain-of-thought thermodynamics training data of sufficient quality to measurably improve domain reasoning in fine-tuned models — without access to large GPU clusters?*

The answer is yes — with an important qualification: improvement is directly proportional to training data coverage of the topic. Fine-tuning improved mean Gemini-judge scores by ~22–33% and reduced hallucination by 50% on well-represented topics. Topics absent from the training data showed no improvement regardless of model size.

This finding directly relates to concurrent peer-reviewed research: Loubet et al. (2025), *Computers and Chemical Engineering*, who benchmarked frontier LLMs on thermodynamics problems without fine-tuning and proposed synthetic verified training data as a path to improvement — which is precisely what this project implements.

---

## Pipeline Architecture

```
5 Open-Access Thermodynamics Textbooks
              │
              ▼
    ┌─────────────────────┐
    │   PDF → Markdown    │  pymupdf4llm
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   Text Chunking     │  3,000 character chunks
    └─────────────────────┘
              │
    ┌─────────┴──────────────────────┐
    ▼                                ▼
┌──────────────────┐      ┌──────────────────────┐
│ Stage 1:         │      │ Stage 2:             │
│ Question         │─────▶│ Answer Generation    │
│ Generation       │      │ (Gemini 2.5 Pro)     │
│ (Gemini 2.5 Pro) │      └──────────────────────┘
└──────────────────┘                │
                                    ▼
                         ┌──────────────────────┐
                         │ Stage 3: Vetting     │
                         │ (Gemini 2.5 Flash)   │
                         └──────────────────────┘
                                    │
                          ┌─────────┴────────┐
                          │ VALID            │ INCORRECT
                          ▼                  ▼
                      Save to JSONL      Discard
```

The three-stage design deliberately separates generation from quality control. Gemini Pro generates; Gemini Flash audits. Flash is faster and cheaper, making it appropriate for the binary VALID/INCORRECT judgement at scale, a deliberate cost/quality trade-off.

A contamination check was added to catch cases where the system prompt leaked into the instruction field, ensuring clean Q&A pairs.

---

## Dataset

| Property | Value |
|---|---|
| Source material | 5 open-access thermodynamics textbooks |
| Total validated entries | 1,805 |
| Average output length | ~2,828 characters |
| Max token length | 1,536 tokens (hardware constraint) |
| Format | JSONL — `instruction` and `output` keys |
| Content | Step-by-step derivations with LaTeX notation |

### Topic Distribution

The dataset was analysed for topic coverage — a key step that revealed the primary driver of fine-tuning performance:

| Topic | Entries | % of Dataset |
|---|---|---|
| Enthalpy | 373 | 20.7% |
| Efficiency | 270 | 15.0% |
| Heat engine | 152 | 8.4% |
| Turbine | 139 | 7.7% |
| Carnot | 130 | 7.2% |
| Compressibility | 100 | 5.5% |
| Refrigerator | 39 | 2.2% |
| Throttling | 28 | 1.6% |
| Free expansion | 25 | 1.4% |
| **COP** | **0** | **0.0%** |

This distribution directly predicted fine-tuned model performance — see Results.

---

## Models & Training

Both models used identical training configuration:

| Hyperparameter | Value | Rationale |
|---|---|---|
| Quantisation | 4-bit NF4 | Fits large models in Colab GPU memory |
| LoRA rank (r) | 16 | Sufficient capacity for domain adaptation |
| LoRA alpha | 32 | Standard 2× rank scaling |
| Target modules | q, k, v, o + gate, up, down (MLP) | MLP layers store factual domain knowledge |
| Effective batch size | 32 | 1 per device × 32 gradient accumulation steps |
| Learning rate | 5e-5 | Stable for domain-specific training |
| Epochs | 2 | Prevents memorisation on small dataset |
| Optimiser | paged_adamw_8bit | Memory-efficient |
| Max sequence length | 1,536 tokens | Hardware constraint |

### Why MLP layers were added

The initial configuration targeted only attention projections (q, v, o). Attention handles context and retrieval; MLP layers store factual associations. Adding gate_proj, up_proj, and down_proj increased trainable parameters 10× and is well-supported for domain knowledge transfer.

### Training Loss

**Llama-3.2-3B-Instruct:**

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 50 | 0.9538 | 0.9988 |
| 100 | 0.9808 | 0.9722 |

**Mistral-7B-Instruct-v0.2:**

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 25 | 0.9109 | 0.8228 |
| 50 | 0.7679 | 0.7654 |
| 75 | 0.7590 | 0.7439 |
| 100 | 0.7374 | 0.7366 |

Mistral converged to a lower final val loss (0.7366) , consistent with its larger parameter count. Neither model showed val loss divergence — no overfitting within the 2-epoch budget.

---

## Evaluation

### Method: Gemini-as-Judge

A Gemini 2.5 Flash judge scored each response against a reference answer:

| Criterion | Max | Description |
|---|---|---|
| Correctness | 4 | Is the core answer/calculation correct? |
| Reasoning | 3 | Is the step-by-step working sound? |
| Clarity | 2 | Is the answer clearly explained? |
| Hallucination penalty | -1 | Deducted for confidently wrong facts |
| **Total** | **9** | |

This approach aligns methodologically with Loubet et al. (2025), who used trained human experts following a comparable rubric.

### Evaluation Design

Questions were split by training data frequency to test the data coverage hypothesis:

**High-frequency topics** (enthalpy ~21%, efficiency ~15%, heat engine ~8%)
**Low-frequency topics** (COP 0%, free expansion 1.4%, throttling 1.6%)

---

## Results

### Primary Finding: Fine-Tuning Improved Performance on Well-Represented Topics

| Model | Mean /9 | Std | Hallucinations |
|---|---|---|---|
| Llama 3B — base | 3.9 | 3.02 | 3.0 avg |
| **Llama 3B — fine-tuned** | **5.3** | **1.44** | **1.5 avg** |
| Mistral 7B — base | 3.5–4.3 | 2.6–3.4 | 3.0 avg |
| **Mistral 7B — fine-tuned** | **5.1–5.5** | **1.0–1.9** | **1.0–2.0 avg** |

Fine-tuning improved mean scores by ~1.5–2 points across both architectures. Standard deviation reduced significantly in both cases — the fine-tuned models are more consistent than base models.

### Secondary Finding: Data Coverage Drives Performance

This is the most important finding of the project.

**High-frequency topic results (Mistral 7B, avg of 2 runs):**

| Question | Topic | Base | Fine-tuned |
|---|---|---|---|
| Q01 | Specific Enthalpy | 1.75 | 4.75 |
| Q02 | Enthalpy + Heating | 0.75 | 5.0 |
| Q03 | Heat Engine Net Work | 3.0 | 6.5 |
| Q04 | Thermal Efficiency | 7.0 | 6.75 |

**Low-frequency topic results (Mistral 7B):**

| Question | Topic | Base | Fine-tuned |
|---|---|---|---|
| Q01 | Refrigerator COP | ~0.5 | ~2.0 |
| Q02 | Free Expansion | ~1.0 | ~0.5 |
| Q03 | Throttling | ~1.0 | ~0.0 |

The contrast is stark. Topics with hundreds of training examples improved meaningfully. Topics with zero or near-zero training examples showed no improvement — or regression, because the model applies learned domain style confidently but incorrectly.

### The Chain-of-Thought Effect

Training data was not just Q&A pairs — every entry was a step-by-step derivation with explicit formula declarations, numbered reasoning chains, and LaTeX notation. The Flash vetter rejected entries that did not meet this standard.

This enforced chain-of-thought format transferred to model behaviour: fine-tuned models produce more structured responses, ramble less, and apply formulas more consistently — even on questions where the final answer is wrong. The model learned *how to reason through a problem* structurally, not just what the answer is.

### Model Size Was Not the Primary Driver

Mistral 7B did not clearly outperform Llama 3B after fine-tuning. Both models showed similar improvement patterns and similar failure modes on underrepresented topics. This suggests that at this dataset scale (1,805 entries), training data coverage matters more than model capacity.

---

## Failure Analysis

### Primary failure mode: Style transfer without knowledge transfer

The fine-tuned model learned the structure and vocabulary of expert thermodynamics responses more thoroughly than the underlying domain principles. On entropy generation problems, the model invented a non-existent formula `S_gen = ΔS / T_avg` with full confidence — correct format, completely wrong physics.

This is a known limitation of fine-tuning on small datasets: surface patterns are learned before deep relationships.

### Secondary failure mode: Dataset coverage gaps

Topics with zero or minimal training examples showed no improvement regardless of model size. COP (0 entries) was the clearest example — both Llama and Mistral consistently failed refrigerator energy balance questions that any thermodynamics textbook would cover in chapter two.

**Implication:** Topic frequency analysis before training is as important as total dataset size. A 1,000-entry dataset with balanced coverage across all thermodynamics topics would likely outperform a 2,000-entry dataset skewed toward a subset.

### Tertiary failure mode: Token truncation

The 1,536 token cap silently truncated longer training answers. Multi-step derivations requiring 2,000+ tokens were cut off mid-calculation. The model learned to produce responses of approximately this length, sometimes stopping before completing the reasoning chain.

### What would fix this

1. **Balanced dataset** — audit topic distribution before training, target underrepresented areas (COP, throttling, free expansion, Joule-Thomson, refrigeration cycles) with equal representation
2. **Larger dataset** — 10,000+ entries to support reliable mathematical generalisation
3. **Remove token cap** — train on GPU with larger memory to allow full-length derivations
4. **Completion-only training** — mask loss on question tokens so the model only learns to predict the answer
5. **Negative examples** — include deliberately incorrect reasoning labelled as invalid to teach the model what wrong looks like

---

## Relation to Published Research

Loubet et al. (2025), *"Using large language models for solving textbook-style thermodynamic problems"*, *Computers and Chemical Engineering* 204, 109333.

This paper benchmarked GPT-3.5, GPT-4, GPT-4o, Llama 3.1 70B, and le Chat on 22 textbook thermodynamics problems without fine-tuning. Key overlapping findings:

- All models perform well on simple problems but deteriorate on complex multi-step reasoning — consistent with ThermoLLM's results
- Llama 3.1's primary error types (wrong equations, context confusion) match the failure modes observed here
- The authors explicitly propose using knowledge-based problem solvers to generate verified training data as a direction for improvement — the three-stage synthetic pipeline in this project is a direct implementation of that proposal, developed independently

---

## Repository Structure

```
thermo-llm-finetuning/
│
├── README.md
├── requirements.txt
│
├── data_pipeline/
│   └── generate_dataset.py       # 3-stage Gemini pipeline
│
├── training/
│   └── finetune.py               # QLoRA training (Llama + Mistral)
│
├── evaluation/
│   └── evaluate.py               # Gemini-as-judge evaluation
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb    # Original Colab: data generation
│   ├── 02_llama_finetuning.ipynb # Original Colab: Llama training
│   └── 03_mistral_finetuning.ipynb # Original Colab: Mistral training
│
├── data/
│   └── thermo_dataset.jsonl      # 1,805 validated Q&A pairs
│
└── results/
    └── (evaluation CSVs go here)
```

---

## Reproducing This Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset (Gemini API key required, CPU only)
```bash
python data_pipeline/generate_dataset.py \
    --pdf your_textbook.pdf \
    --output data/thermo_dataset.jsonl \
    --api_key YOUR_GEMINI_API_KEY
```
Pre-generated dataset (1,805 entries) available in `data/thermo_dataset.jsonl`.

### 3. Fine-tune (Colab GPU required)
```bash
# Llama 3B
python training/finetune.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --output ./outputs/llama-thermo \
    --hf_token YOUR_HF_TOKEN

# Mistral 7B
python training/finetune.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --output ./outputs/mistral-thermo \
    --hf_token YOUR_HF_TOKEN
```

### 4. Evaluate
```bash
# Llama (merged model)
python evaluation/evaluate.py \
    --base_model meta-llama/Llama-3.2-3B-Instruct \
    --finetuned_model ./outputs/llama-merged \
    --hf_token YOUR_HF_TOKEN \
    --gemini_key YOUR_GEMINI_KEY

# Mistral (adapter only — avoids 14GB merge)
python evaluation/evaluate.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter_path ./outputs/mistral-thermo \
    --hf_token YOUR_HF_TOKEN \
    --gemini_key YOUR_GEMINI_KEY
```

---

## Key Takeaways

1. **Fine-tuning on synthetic chain-of-thought data works.** Mean scores improved ~1.5–2 points and hallucination rates halved across two model architectures.

2. **Data coverage is the primary driver of performance, not model size.** Mistral 7B did not clearly outperform Llama 3B after fine-tuning on the same dataset. Topics with zero training examples failed regardless of model capacity.

3. **Chain-of-thought format transfers.** Training on structured step-by-step derivations reduced rambling and improved reasoning structure even when final answers were incorrect.

4. **Topic frequency analysis is essential.** Auditing dataset distribution before training would have revealed the COP coverage gap and allowed targeted data generation to fix it.

5. **This aligns with peer-reviewed findings.** The observed failure modes and proposed solutions map directly onto Loubet et al. (2025), validating both the experimental approach and the conclusions independently.

---

## References

Loubet, R., Zittlau, P., Vollmer, L., Hoffmann, M., Fellenz, S., Jirasek, F., Leitte, H., & Hasse, H. (2025). Using large language models for solving textbook-style thermodynamic problems. *Computers and Chemical Engineering*, 204, 109333. https://doi.org/10.1016/j.compchemeng.2025.109333

---

**Author:** Egameh Omokagbo | [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)
