import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import torch
import numpy as np
from tqdm import tqdm
from peft import PeftModel, prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
import transformers
import os
from ppl import perplexity
from utils import get_prompt, get_bnb_config

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

print("Loading data.")
train_data = load_json("./train.json")
public_test = load_json("./public_test.json")

print("Loading model.")
model_name = "zake7749/gemma-2-2b-it-chinese-kyara-dpo"
bnb_config = get_bnb_config()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0})

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=12,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

examples = train_data[:20]
examples_prompt = "\n".join([f"範例 {i + 1}: {ex['instruction']} {ex['output']}" for i, ex in enumerate(examples)])

# Add examples to "instruction"
for item in train_data:
    item["instruction"] = f"{examples_prompt}\n指令: {item['instruction']}"

for item in public_test:
    item["instruction"] = f"{examples_prompt}\n指令: {item['instruction']}"

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(public_test)

def preprocess_function(examples):
    processed_instructions = [get_prompt(instr) for instr in examples["instruction"]]

    # Tokenize processed instructions (inputs) and outputs separately
    tokenized_instructions = tokenizer(processed_instructions, truncation=True, padding=False)
    tokenized_outputs = tokenizer(examples["output"], truncation=True, padding=False)
    
    input_ids = []
    attention_masks = []
    output_masks = []
    
    max_length = 2048  # 定義最大長度
    data_size = len(examples["instruction"])
    
    for i in range(data_size):
        # Add special tokens: bos_token_id at the start of instruction and eos_token_id at the end of output
        instruction_input_ids = [tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + [tokenizer.eos_token_id]

        # Concatenate instruction and output ids
        concatenated_input_ids = instruction_input_ids + output_input_ids
        concatenated_attention_mask = [1] * len(concatenated_input_ids)

        # Create output mask to differentiate between instruction and output
        output_mask = [0] * len(instruction_input_ids) + [1] * len(output_input_ids)

        # Ensure input length does not exceed max_length
        concatenated_input_ids = torch.tensor(concatenated_input_ids[:max_length])
        concatenated_attention_mask = torch.tensor(concatenated_attention_mask[:max_length])
        output_mask = torch.tensor(output_mask[:max_length])

        input_ids.append(concatenated_input_ids)
        attention_masks.append(concatenated_attention_mask)
        output_masks.append(output_mask)
    
    # Return the processed inputs
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "output_mask": output_masks
    }

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

os.environ["WANDB_DISABLED"] = "true"

# def compute_metrics(eval_preds):
#     logits, labels, output_masks = eval_preds
#     if isinstance(logits, tuple):
#         logits = logits[0]

#     # Move predictions and masks to CPU to save GPU memory
#     predictions = torch.tensor(logits).cpu()
#     labels = torch.tensor(labels).cpu()
#     output_masks = torch.tensor(output_masks, dtype=torch.bool).cpu()

#     # Use mask to filter out instruction tokens
#     masked_predictions = predictions[output_masks]
#     masked_labels = labels[output_masks]

#     # Compute perplexity batch-wise to reduce memory load
#     batch_size = 512  # Adjust batch size based on available memory
#     losses = []
#     with torch.no_grad():
#         for i in range(0, masked_predictions.size(0), batch_size):
#             batch_logits = masked_predictions[i : i + batch_size]
#             batch_labels = masked_labels[i : i + batch_size]

#             shift_logits = batch_logits[..., :-1, :].contiguous()
#             shift_labels = batch_labels[..., 1:].contiguous()
#             loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             losses.append(torch.exp(loss).item())

#     # Calculate mean perplexity
#     mean_perplexity = sum(losses) / len(losses)
#     return {"mean_perplexity": mean_perplexity}
   
# Define training arguments
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,  
    gradient_accumulation_steps=4,
    warmup_steps=500,
    max_steps=2500,
    learning_rate=3e-5,
    fp16=True,
    logging_steps=500,
    evaluation_strategy="steps",  
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    # metric_for_best_model="perplexity",  # 依據 perplexity 評估最佳模型
    # greater_is_better=False,  # perplexity 越小越好
    output_dir="outputs",
    optim="paged_adamw_8bit",
    weight_decay=0.01
)

# Define Trainer
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)

model.config.use_cache = False
print("Start training.")
trainer.train()
model.save_pretrained("adapter_checkpoint")
print("Checkpoint saved.")

## ------ Perform inference on public testing data
adapter_checkpoint = "./adapter_checkpoint"  
base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": 0})

# Load the fine-tuned adapter checkpoint
model = PeftModel.from_pretrained(base_model, adapter_checkpoint)
model.eval()  # Set model to evaluation mode
model.config.use_cache = False

print("Start performing inference on public testing data.")
results = perplexity(model, tokenizer, public_test)
print(f"\nMean Perplexity: {results['mean_perplexity']}")
