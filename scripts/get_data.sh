#!/bin/bash
# Download the benchmark (VerilogEval v2, MIT licence) and the training-pool source
# (VeriThoughts consistent-instruction set, MIT licence). About 20 MB in total.
set -e
cd "$(dirname "$0")/.."
mkdir -p data
[ -d verilog-eval ] || git clone -q --depth 1 https://github.com/NVlabs/verilog-eval.git
# -f fails on HTTP errors; --retry and -C - resume a dropped transfer instead of saving half a file.
# --http1.1 because Hugging Face's HTTP/2 stream kept dropping mid-file on a slow link.
curl -sSfL --http1.1 --retry 5 --retry-all-errors -C - -o data/vt_consistent.json \
  "https://huggingface.co/datasets/wilyub/VeriThoughtsTrainSetConsistentInstruction/resolve/main/lf_complete_trainset_instructions.jsonl"
python3 -c "import json; print(len(json.load(open('data/vt_consistent.json'))), 'pool problems downloaded')"
