# Rules for Claude Code in this repository

This is a learning project. I must be able to explain every line in an interview.

1. Explain every change in plain language before making it, and keep changes small.
2. After any change to `rtlrl/`, run `python -m pytest -q` and show me the result.
3. Never weaken the grader to make a number go up. If a test fails, tell me why.
4. Never put VerilogEval problems, HDLBits-based data or anything in `fresh/` into a training file.
5. Never change evaluation settings between a baseline and a trained model: same prompt
   (`rtlrl/prompt.py`), n=20, temperature 0.85, top_p 0.95, max 1024 tokens, same Icarus 12.
6. Every number I report must come from `results/*.graded.csv` or `scripts/compare.py`.
   Never round up. Say "not proven" when the confidence interval includes zero.
7. Every training run gets a LOG.md entry: date, git commit, settings, numbers.
8. Ask before deleting data or results.
