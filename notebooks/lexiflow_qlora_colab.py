# %% [markdown]
# # LexiFlow QLoRA Prototype (Google Colab)
# Run this file in Colab (T4 GPU runtime).

# %%
import os
import shutil
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/Vivek-afk81/mini-gpt-research-framework.git"
REPO_DIR = Path("/content/mini-gpt-research-framework")
ADAPTER_DIR = REPO_DIR / "checkpoints" / "qlora"


def run(cmd):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


# %% [markdown]
# ## 1) Verify GPU runtime

# %%
run("nvidia-smi")


# %% [markdown]
# ## 2) Clone (or reuse) repository

# %%
if REPO_DIR.exists():
    print(f"Using existing repo at: {REPO_DIR}")
else:
    run(f"git clone {REPO_URL} {REPO_DIR}")

os.chdir(REPO_DIR)
print(f"Working directory: {Path.cwd()}")


# %% [markdown]
# ## 3) Install dependencies

# %%
run("python -m pip install --upgrade pip")
run("pip install -r requirements.txt -r requirements-qlora.txt")


# %% [markdown]
# ## 4) Prepare training data

# %%
run("python src/finetune/dataset.py")
run("python - <<'PY'\nimport json\np='data/finetuning/train_chat.jsonl'\nwith open(p,encoding='utf-8') as f:\n    first=json.loads(next(line for line in f if line.strip()))\nprint('messages roles:', [m['role'] for m in first['messages']])\nprint('sample user:', first['messages'][1]['content'])\nPY")


# %% [markdown]
# ## 5) Train QLoRA adapter (Qwen 1.5B default)

# %%
run(
    "python src/finetune/train_qlora.py "
    "--model Qwen/Qwen2.5-1.5B-Instruct "
    "--data data/finetuning/train_chat.jsonl "
    "--output checkpoints/qlora "
    "--epochs 3 "
    "--batch-size 4 "
    "--lr 2e-4 "
    "--max-length 512 "
    "--lora-r 16"
)


# %% [markdown]
# ## 6) Quick post-training test

# %%
run("python scripts/test_finetuned.py --max-new-tokens 80")


# %% [markdown]
# ## 7) Zip adapter and download

# %%
zip_path = REPO_DIR / "lexiflow-qlora.zip"
if zip_path.exists():
    zip_path.unlink()

shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=REPO_DIR, base_dir="checkpoints/qlora")
print(f"Created: {zip_path}")

try:
    from google.colab import files  # type: ignore
    files.download(str(zip_path))
except Exception:
    print("If not running in Colab UI, manually download:", zip_path)
