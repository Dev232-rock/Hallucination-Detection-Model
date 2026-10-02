# Hallucination Detection Model

A token-level hallucination detection system that trains lightweight **linear probes** on transformer hidden states. The probe learns to classify individual tokens as *hallucinated* or *factual* by monitoring internal activations of a large language model — no external knowledge base required.

---

## Architecture

```
ProbedModel
├── Base LLM  (e.g. LLaMA-3.1-8B-Instruct) — partially frozen
├── LoRA adapters  — fine-tuned on selected layers
└── ProbeHead  — linear classifier on top of hidden states at layer L
```

Multi-layer probing is also supported via `MultiLayerProbeHead`, which aggregates activations from several layers with learned weights.

---

## Project Structure

```
Hallucination-Detection-Model/
├── probe/
│   ├── __init__.py             # Package exports
│   ├── types.py                # AnnotatedSpan, ProbingItem dataclasses
│   ├── config.py               # ProbeConfig, TrainingConfig, EvaluationConfig
│   ├── dataset.py              # TokenizedProbingDataset (token-level labels)
│   ├── dataset_converters.py   # HF dataset → ProbingItem converters
│   ├── model.py                # ProbeHead, MultiLayerProbeHead, ProbedModel
│   ├── train.py                # Training loop entry point
│   ├── evaluate.py             # Evaluation entry point
│   └── inference.py            # HallucinationDetector API + CLI
├── utils/
│   ├── hooks.py                # Forward hook context manager
│   ├── metrics.py              # AUC, F1, ROC curves
│   ├── model_utils.py          # Model loading, LoRA setup
│   ├── probe_loader.py         # HuggingFace upload/download
│   ├── tokenization.py         # Binary-search token span finder
│   ├── string_utils.py         # ROUGE-based fuzzy matching
│   ├── parsing.py              # JSON/Pydantic response parser
│   └── files_utlis.py          # JSONL / JSON / YAML I/O
├── configs/
│   ├── train_llama3_8b.yaml    # Sample training config
│   └── eval_llama3_8b.yaml     # Sample evaluation config
└── value_head_probes/          # Saved probe checkpoints (auto-created)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch transformers peft datasets huggingface_hub \
            scikit-learn matplotlib jaxtyping termcolor tqdm \
            pydantic pydantic-core rouge-score wandb
```

### 2. Train a probe

```bash
python -m probe.train --config configs/train_llama3_8b.yaml
```

### 3. Evaluate

```bash
python -m probe.evaluate --config configs/eval_llama3_8b.yaml
```

### 4. Inference (Python API)

```python
from probe.inference import HallucinationDetector

detector = HallucinationDetector.from_pretrained(
    probe_id="llama3_1_8b_lora_lambda_kl=0.5",
    load_from="disk",   # or "hf" to pull from HuggingFace
)

result = detector.detect(
    prompt="Who invented the telephone?",
    completion="Alexander Graham Bell invented the telephone in 1876, "
               "though many historians credit Antonio Meucci with the original design in 1854."
)

print(result)
# → Hallucinated: False  (max score: 0.12)

# Highlight suspicious spans
print(detector.highlight(result))
```

### 5. Inference (CLI)

```bash
python -m probe.inference \
    --probe_id llama3_1_8b_lora_lambda_kl=0.5 \
    --prompt "Who invented the telephone?" \
    --completion "Bell invented it in 1976 during the industrial revolution." \
    --threshold 0.5
```

---

## Supported Datasets

| Dataset ID  | HuggingFace Repo         | Notes                          |
|-------------|--------------------------|--------------------------------|
| `ragtruth`  | `wandb/RAGTruth`         | RAG hallucination benchmark    |
| `felm`      | *(custom)*               | Factual error localisation     |
| `halueval`  | *(custom)*               | Sentence-level hallucination   |
| `generic`   | any                      | Needs `prompt/completion/spans` columns |

Add a new dataset by implementing a `prepare_<name>(row: dict) -> ProbingItem` function in [`probe/dataset_converters.py`](probe/dataset_converters.py).

---

## Supported Models

| Model                                        | Layers |
|----------------------------------------------|--------|
| `meta-llama/Meta-Llama-3.1-8B-Instruct`      | 32     |
| `meta-llama/Meta-Llama-3.1-70B-Instruct`     | 80     |
| `meta-llama/Llama-3.3-70B-Instruct`          | 80     |
| `google/gemma-2-2b-it`                       | 26     |
| `google/gemma-2-9b-it`                       | 42     |
| `google/gemma-2-27b-it`                      | 46     |
| `Qwen/Qwen2.5-7B-Instruct`                   | 28     |
| `Qwen/Qwen2.5-14B-Instruct`                  | 48     |
| `mistralai/Mistral-Small-24B-Instruct-2501`  | 40     |

---

## License

MIT
