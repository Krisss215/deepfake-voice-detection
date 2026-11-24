#!/usr/bin/env python3
"""Downloads the two frozen self-supervised speech models used by approach B (~380 MB each)
into data/models/. Run on your own computer: pip install huggingface_hub"""
import sys
from huggingface_hub import snapshot_download

out = sys.argv[1] if len(sys.argv) > 1 else "data/models"
for repo in ["facebook/wav2vec2-base", "microsoft/wavlm-base-plus"]:
    name = repo.split("/")[1]
    print("downloading", repo)
    snapshot_download(repo, local_dir=f"{out}/{name}",
                      allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"])
print("done ->", out)
