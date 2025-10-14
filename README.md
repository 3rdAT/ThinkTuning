# ThinkTuning

This repository consists the code for ThinkTuning, a GRPO based training approach.
## Quick Setup (Recommended)

```bash
# Complete setup with one command
make setup

# Activate the environment
source .venv/bin/activate
```

## Manual Setup (Alternative)

```bash
uv venv
source .venv/bin/activate  # On Linux/Mac
# Windows: .venv\Scripts\activate

# Install dependencies
uv pip install packaging wheel setuptools
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install -e .

# Setup verl
cd verl
uv pip install -e ".[vllm]"
cd ..

uv pip install msgspec
uv pip install deepspeed
uv pip install trl
uv pip install nltk unidecode sacremoses
uv pip install math-verify

```
