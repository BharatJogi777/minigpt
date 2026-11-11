# MiniGPT from Scratch

An educational decoder-only GPT-style language model implemented with PyTorch. The model starts from random weights and learns next-character prediction from a text corpus. It is a learning project, not a ChatGPT-scale foundation model.

## Apple Silicon / 8 GB RAM

Defaults are intentionally small: 4 Transformer blocks, 4 attention heads, 128 hidden dimensions, context length 128, batch size 8. Start with these settings and reduce them if memory pressure occurs. MPS is selected when available; otherwise CPU is used. Performance and memory use depend on macOS and PyTorch versions.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10+ is required. On Apple Silicon, install a PyTorch build with MPS support.

## Check device

```bash
python scripts/check_device.py
```

## Prepare data

Replace or extend `data/input.txt` with UTF-8 text you have permission to use. The included corpus is only a tiny smoke-test fixture; it is not sufficient to train a useful language model. Keep secrets and personal data out of the dataset.

## Train

```bash
python -m minigpt.train --data data/input.txt --steps 2000
```

Training writes `artifacts/vocab.json` and `checkpoints/latest.pt`. Try a smaller configuration if needed:

```bash
python -m minigpt.train --steps 500 --batch-size 4 --block-size 96 --n-embd 96 --n-layer 2 --n-head 2
```

## Generate

```bash
python -m minigpt.generate --prompt "The " --max-new-tokens 100
```

## API

Install the dependencies, train a checkpoint, then run:

```bash
uvicorn minigpt.api:app --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/generate \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt":"The ","max_new_tokens":80}'
```

The API loads the checkpoint at startup. Keep it bound to localhost unless you add authentication and deployment security controls.

## Architecture

- Deterministic character-level tokenizer (BPE is a planned extension)
- Token and learned positional embeddings
- Causal multi-head self-attention
- Pre-layer-normalized Transformer blocks with GELU feed-forward networks
- Tied token embedding / language-model output weights
- Next-token cross-entropy objective and AdamW optimizer

## Tests

```bash
pytest -q
```

## Docker

After training locally, `docker compose up --build` starts the inference API and mounts the saved artifacts read-only. The container runs CPU inference; it does not provide Apple MPS GPU access.

## Known limitations / roadmap

The model is tiny and the starter corpus is deliberately small. It will produce limited, often incoherent text. It does not possess broad factual knowledge. Next milestones include BPE tokenization, larger curated datasets, validation perplexity tracking, gradient accumulation, more robust checkpointing, and experiment reproducibility.
