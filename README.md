# RTL-RL starter

Teach a small open AI model to write correct Verilog, using free simulators as the judge,
on free Kaggle GPUs. This folder goes with `Kumud_Starting_Project_Roadmap.docx`, which
explains every step in plain language. Rename the project before you publish it
("VeriRL" is already the name of a published paper).

## What is in here

| Path | What it does |
|---|---|
| `rtlrl/extract.py` | Pulls the Verilog out of a model's reply |
| `rtlrl/safety.py` | Rejects designs that use `$display`, `$finish`, files, `` `include `` and similar tricks |
| `rtlrl/tbgen.py` | Turns any reference design into an automatic random-input testbench |
| `rtlrl/trace_grader.py` | Simulates a design and compares its outputs to the reference's recording |
| `rtlrl/ve_grader.py` | Grades exactly like VerilogEval v2 (same flags, same pass rule) |
| `rtlrl/reward.py` | The reward function TRL calls during GRPO |
| `rtlrl/stats.py` | pass@k and the paired bootstrap confidence interval |
| `rtlrl/decontam.py` | Finds training problems that look like test problems |
| `scripts/` | One script per step: build the pool, split, generate, grade, compare, train, audit |
| `tests/` | 18 tests that try to fool the grader. Run them after every change |
| `kaggle/` | Three notebooks to import into Kaggle (File > Import Notebook) |
| `fresh/` | Your own test problems. One example is included; write 29 more |

## Quick start on your Mac

```bash
export PATH="$HOME/oss-cad-suite/bin:$PATH"
python3 -m venv .venv
.venv/bin/pip install -r requirements-mac.txt
.venv/bin/python -m pytest -q
```

You should see `18 passed`. (Use the `export PATH` line, not `source ~/oss-cad-suite/environment`:
the environment script also swaps in the suite's own python3, and then `.venv` is created without pip.)

Then build the training pool:

```bash
./scripts/get_data.sh
.venv/bin/python scripts/build_pool.py --source data/vt_consistent.json --verilogeval verilog-eval/dataset_spec-to-rtl --fresh fresh --out data/pool.jsonl
.venv/bin/python scripts/split_pool.py --pool data/pool.jsonl --heldout 300 --candidates 3000
```

Upload `data/candidates.jsonl` and `data/heldout.jsonl` to Kaggle as a private dataset
named `rtlrl-pool`, then run the notebooks in `kaggle/` in order.

## Facts checked on 15 October 2026

- VerilogEval v2 needs Icarus Verilog **12**. The Icarus in OSS CAD Suite (14, development
  version) fails every VerilogEval testbench, so use it only for the tests and the pool.
- With Icarus 12, **152 of 156** VerilogEval reference solutions pass the official rule.
  Four cannot pass because of the benchmark itself: Prob099 (testbench and reference use
  different port names) and Prob082, Prob141, Prob156 (the testbench hits its own time limit
  and prints TIMEOUT).
- VerilogEval's 156 problems come from HDLBits, so never train on HDLBits-based data.
- Licences: VerilogEval (MIT), VeriThoughts (MIT).

## Results (fill this in as you go)

| Model | VerilogEval v2 pass@1 (n=20) | Gain vs base, 95% CI | Held-out pass@1 | Fresh pass@1 | Date |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct (base) | | | | | |
