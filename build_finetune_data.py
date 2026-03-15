import os
import json
import pandas as pd

def build_finetune_data(tsv_path, output_jsonl, max_samples=5000):
    """
    基于现有的 query 数据，构建 SFT 数据集。
    按照 Filter-ANN 的需求，模型不仅仅需要做意图泛化改写，
    必须严格输出包含 `rewritten` 和 `attributes` (如类目、颜色) 的 JSON 结构。
    """
    print(f"Loading queries from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t', header=0)
    
    unique_queries = df['query'].dropna().unique()
    print(f"Found {len(unique_queries)} unique queries.")
    
    samples = unique_queries[:max_samples]
    formatted_data = []
    
    # 提示词系统模板，强制要求 JSON 输出
    system_prompt = (
        "你是一个电商搜索意图理解专家。请同时完成以下两项任务：\n"
        "1. 对用户的搜索词进行泛化与改写，生成更详细的搜索短语。\n"
        "2. 从用户的搜索词中提取出具体的商品属性（如：category，color）。如果未提及则留空。\n"
        "请严格只输出一个合法的 JSON 字符串，格式为：\n"
        "{\"rewritten\": \"...\", \"attributes\": {\"category\": \"...\", \"color\": \"...\"}}"
    )
    
    for q in samples:
        q_str = str(q)
        
        # --- 伪造高质量的 Teacher 标注结果 ---
        # 实际开发中通过调用拥有丰富知识的大模型（例如经过微调前的 Qwen-Max）来自动化打标
        # 例如 q_str == "treble popular reed" (高音流行簧片) 
        pseudo_response = json.dumps({
            "rewritten": f"寻找音质极佳的{q_str}配件，适合乐器演奏的优质簧片",
            "attributes": {
                "category": "musical instrument accessories",
                "color": ""
            }
        }, ensure_ascii=False)
        
        record = {
            "type": "chatml",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": q_str
                },
                {
                    "role": "assistant",
                    "content": pseudo_response
                }
            ],
            "source": "kdd_structured_query"
        }
        formatted_data.append(record)
        
    print(f"Saving {len(formatted_data)} records to {output_jsonl}...")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("✅ Structured SFT fine-tuning dataset built successfully.")


if __name__ == '__main__':
    TSV_FILE = "data/sample/train.sample.tsv"
    OUTPUT_FILE = "qwen_structured_query_sft.jsonl"
    
    if os.path.exists(TSV_FILE):
        build_finetune_data(TSV_FILE, OUTPUT_FILE)
    else:
        print(f"⚠️ 找不到数据文件 {TSV_FILE}")
