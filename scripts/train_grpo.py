"""GRPO training on the learnable-zone problems.

Smoke test on your Mac (tiny random model, CPU, 2 steps; checks the plumbing only):
  python scripts/train_grpo.py --smoke --pool data/pool_small.jsonl

On Kaggle (T4 GPU), after running the install cell from Unsloth's GRPO notebook:
  python scripts/train_grpo.py --model unsloth/Qwen2.5-Coder-1.5B-Instruct \
      --pool data/zone.jsonl --steps 300 --out /kaggle/working/grpo_run1
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--model", default="unsloth/Qwen2.5-Coder-1.5B-Instruct")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--generations", type=int, default=8)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--seed", type=int, default=3407)
ap.add_argument("--out", default="outputs/grpo")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

if not args.smoke:
    from unsloth import FastLanguageModel  # must be imported before trl

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtlrl.pool import load_pool
from rtlrl.prompt import make_messages
from rtlrl.reward import verilog_reward

rows = load_pool(args.pool)  # re-records references with this machine's simulator
dataset = Dataset.from_list([
    {"prompt": make_messages(r["spec"]), "tb": r["tb"], "ref_trace": r["ref_trace"]} for r in rows
]).shuffle(seed=args.seed)
print(f"{len(dataset)} training problems")

common = dict(
    output_dir=args.out,
    learning_rate=args.lr,
    num_generations=args.generations,
    per_device_train_batch_size=args.generations,  # one problem's whole group per micro-batch
    gradient_accumulation_steps=4,                  # 4 problems per optimizer step
    max_completion_length=768,
    mask_truncated_completions=True,                # cut-off answers teach nothing
    temperature=1.0,
    max_steps=args.steps,
    logging_steps=1,
    save_steps=50,
    report_to="none",
    seed=args.seed,
)

if args.smoke:
    from peft import LoraConfig
    model = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = None
    peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    config = GRPOConfig(**{**common, "num_generations": 4, "per_device_train_batch_size": 4,
                           "gradient_accumulation_steps": 1, "max_completion_length": 32,
                           "max_prompt_length": 512, "max_steps": 2, "save_steps": 1000,
                           "use_cpu": True})
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, load_in_4bit=False,
        fast_inference=True, max_lora_rank=args.rank, gpu_memory_utilization=0.85)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed)
    peft_config = None
    # the same optimizer settings as Unsloth's Kaggle GRPO notebook
    config = GRPOConfig(**common, use_vllm=True, max_prompt_length=1024, optim="adamw_8bit",
                        warmup_ratio=0.1, lr_scheduler_type="cosine", weight_decay=0.001,
                        adam_beta1=0.9, adam_beta2=0.99, max_grad_norm=0.1)

trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[verilog_reward],
                      args=config, train_dataset=dataset, peft_config=peft_config)
trainer.train()
if not args.smoke:
    model.save_lora(os.path.join(args.out, "lora"))
    model.save_pretrained_merged(os.path.join(args.out, "merged"), tokenizer, save_method="merged_16bit")
