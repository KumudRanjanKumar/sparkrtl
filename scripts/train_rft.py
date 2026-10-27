"""Rejection-sampling fine-tuning (RFT): train the model on its own correct answers.

Smoke test on your Mac (tiny random model, CPU, 2 steps):
  python scripts/train_rft.py --smoke --data data/rft.jsonl
On Kaggle (T4 GPU), after the Unsloth install cell:
  python scripts/train_rft.py --model unsloth/Qwen2.5-Coder-1.5B-Instruct --data data/rft.jsonl \
      --out /kaggle/working/rft_run1
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="unsloth/Qwen2.5-Coder-1.5B-Instruct")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--seed", type=int, default=3407)
ap.add_argument("--out", default="outputs/rft")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

if not args.smoke:
    from unsloth import FastLanguageModel  # must be imported before trl

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtlrl.prompt import make_messages

rows = [json.loads(line) for line in open(args.data)]
# prompt/completion format: the loss is computed on the answer only, not on the question
dataset = Dataset.from_list([
    {"prompt": make_messages(r["spec"]), "completion": [{"role": "assistant", "content": r["answer"]}]}
    for r in rows
]).shuffle(seed=args.seed)
print(f"{len(dataset)} training examples")

common = dict(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
              per_device_train_batch_size=4, gradient_accumulation_steps=4, logging_steps=5,
              save_strategy="no", report_to="none", seed=args.seed)

if args.smoke:
    from peft import LoraConfig
    model, tokenizer = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", None
    peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    config = SFTConfig(**{**common, "max_steps": 2, "use_cpu": True})
else:
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=2048,
                                                         load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed)
    peft_config = None
    config = SFTConfig(**common, optim="adamw_8bit", warmup_ratio=0.05, lr_scheduler_type="linear",
                       max_length=2048)

trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset,
                     args=config, peft_config=peft_config)
trainer.train()
if not args.smoke:
    model.save_pretrained_merged(os.path.join(args.out, "merged"), tokenizer, save_method="merged_16bit")
