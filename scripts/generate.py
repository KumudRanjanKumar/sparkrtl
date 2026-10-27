"""Generate answers with vLLM on the Kaggle GPU. Same prompt format for every model.

VerilogEval, 20 samples per problem (the published pass@1 setting):
  python scripts/generate.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --verilogeval verilog-eval/dataset_spec-to-rtl \
      --n 20 --temperature 0.85 --top-p 0.95 --out results/base_ve.jsonl
Training pool, 8 samples per problem (to find the learnable zone):
  python scripts/generate.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --pool data/candidates.jsonl \
      --n 8 --temperature 1.0 --top-p 1.0 --out results/base_pool.jsonl
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtlrl.prompt import make_messages

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="Hugging Face name or a local merged-model folder")
ap.add_argument("--verilogeval", help="folder dataset_spec-to-rtl")
ap.add_argument("--pool", help="a pool .jsonl file")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--temperature", type=float, default=0.85)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

if args.verilogeval:
    stems = sorted(p[:-len("_prompt.txt")] for p in glob.glob(os.path.join(args.verilogeval, "*_prompt.txt")))
    problems = [(os.path.basename(s), open(s + "_prompt.txt").read()) for s in stems]
else:
    problems = [(r["id"], r["spec"]) for r in map(json.loads, open(args.pool))]

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tok = AutoTokenizer.from_pretrained(args.model)
# enable_thinking=False turns off Qwen3's long "thinking" mode; other templates ignore it
prompts = [tok.apply_chat_template(make_messages(spec), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) for _, spec in problems]
llm = LLM(model=args.model, dtype="half", gpu_memory_utilization=0.9, max_model_len=4096, seed=args.seed)
params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

t = time.time()
outs = llm.generate(prompts, params)
secs = time.time() - t
tokens = sum(len(c.token_ids) for o in outs for c in o.outputs)
unique = sum(len({c.text for c in o.outputs}) for o in outs) / len(outs)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    for (pid, _), o in zip(problems, outs):
        f.write(json.dumps({"id": pid, "answers": [c.text for c in o.outputs]}) + "\n")
print(f"{len(problems)} problems x {args.n} answers in {secs:.0f} s; "
      f"{tokens / secs:.0f} generated tokens/s; {tokens / (len(problems) * args.n):.0f} tokens per answer; "
      f"{unique:.1f} different answers per problem on average")
