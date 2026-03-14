import os
import json
import pandas as pd

def build_finetune_data(tsv_path, output_jsonl, max_samples=5000):
    """
    基于现有的 query 数据，利用 LLM (大模型 API) 或人工标注作为 Teacher 
    来批量构建“查询改写”指令微调（SFT）数据集，用于微调像 Qwen3-0.6B 这样的小参数模型。
    """
    print(f"Loading queries from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t', header=0)
    
    # 获取唯一的 querys，避免重复计算
    unique_queries = df['query'].dropna().unique()
    print(f"Found {len(unique_queries)} unique queries.")
    
    samples = unique_queries[:max_samples]
    
    # 构造指令与输入对
    formatted_data = []
    
    # 【注意】实际生产中，这里的 `rewritten_query` 应该通过调用诸如 Qwen-Max/GPT-4 等强大模型，
    # 或者是现有的 `rag_pipeline.py` 中的意图提取函数批量生成。这里为了提供框架，我们对它们进行模拟：
    for q in samples:
        q_str = str(q)
        
        # 伪改写逻辑（演示作用）：提取实体并补全属性
        # 比如 "红色 连衣裙" -> "材质舒适的亮红色短袖连衣裙，适合夏季穿着且具备收腰设计"
        pseudo_rewritten = f"包含核心特征'{q_str}'的优质商品，具备相关属性与精良款式设计"
        
        # 组装为标准的 ChatML / Qwen 训练格式
        record = {
            "type": "chatml",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个电商搜索意图理解专家。请对用户的搜索词进行泛化与改写，提取核心实体与特征属性（材质、款式、颜色、风格等），并生成一段结构清晰的搜索短语，以提升增强检索召回率。"
                },
                {
                    "role": "user",
                    "content": q_str
                },
                {
                    "role": "assistant",
                    "content": pseudo_rewritten
                }
            ],
            "source": "kdd_multimodal_dataset"
        }
        formatted_data.append(record)
        
    print(f"Saving {len(formatted_data)} records to {output_jsonl}...")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("✅ Fine-tuning dataset built successfully.")


if __name__ == '__main__':
    # 你需要准备好 train.sample.tsv 文件路径
    TSV_FILE = "multimodal_train_sampleset/train.sample.tsv"
    OUTPUT_FILE = "qwen_query_expansion_sft.jsonl"
    
    if os.path.exists(TSV_FILE):
        build_finetune_data(TSV_FILE, OUTPUT_FILE)
    else:
        print(f"⚠️ 找不到数据文件 {TSV_FILE}，请先确保数据就绪！这里仅展示数据集构建结构。")
