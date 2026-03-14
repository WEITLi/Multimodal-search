# 这是一个利用 PEFT 库对 Qwen3-0.6B 等小模型进行 LoRA 微调的脚本示例
# 用于学习 query 扩张与重写的意图提取
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def format_data_for_qwen(example, tokenizer, max_length=512):
    # 构建 Qwen 的 Chat 格式 (也可以使用 tokenizer.apply_chat_template)
    messages = example['messages']
    
    # 抽取 user 和 assistant
    user_msg = next((m['content'] for m in messages if m['role'] == 'user'), "")
    assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), "")
    
    # ChatML Format 举例:
    # <|im_start|>system\nYou are an expert...<|im_end|>\n<|im_start|>user\nquery...<|im_end|>\n<|im_start|>assistant\nexpanded...<|im_end|>
    # Qwen-2.5 常用此格式
    prompt = f"<|im_start|>system\n你是一个电商意图理解专家。<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(assistant_msg + "<|im_end|>\n", add_special_tokens=False)
    
    # Pad & Truncate
    full_ids = (input_ids + target_ids)[:max_length]
    labels = ([-100] * len(input_ids) + target_ids)[:max_length]

    # padding
    if len(full_ids) < max_length:
        pad_len = max_length - len(full_ids)
        full_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1 if i != tokenizer.pad_token_id else 0 for i in full_ids]}


def train_lora(model_id="Qwen/Qwen2.5-0.5B-Instruct", data_path="qwen_query_expansion_sft.jsonl", output_dir="./qwen_query_lora"):
    print(f"=====================================")
    print(f"       Qwen-0.6B LoRA SFT")
    print(f"=====================================")
    print(f"Loading Model {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 这里以 bfloat16 为例加载模型，生产中可能应用 bitsandbytes 4bit 
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # 准备 PEFT / LoRA (低秩自适应)
    print("Preparing LoRA Config...")
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model) # 如果是量化模型启用
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 准备 Dataset
    print(f"Loading Dataset {data_path}...")
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    print("Tokenizing data...")
    tokenized_dataset = dataset.map(
        lambda x: format_data_for_qwen(x, tokenizer), 
        batched=False, 
        remove_columns=dataset.column_names
    )

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        fp16=False,
        bf16=True,  # 确保你的 GPU 支持 BFloat16，如 3090/4090 或 A100
        report_to="none"
    )

    # 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
    )
    
    print("Starting Training...")
    trainer.train()
    
    # 保存 Adapter
    print(f"Saving LoRA adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Training finished.")


if __name__ == '__main__':
    train_lora()
