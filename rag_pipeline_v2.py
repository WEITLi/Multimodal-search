import os
import json
import base64
import requests
from pymilvus import MilvusClient
from openai import OpenAI

# ======= 基础配置 =======
session = requests.Session()
session.trust_env = False

DB_PATH    = "/root/qwen3-vl/Qwen3-VL-Embedding-2B/milvus_v2_filter_ann.db"
COLLECTION = "qwen3_vl_images_with_attributes"
IMAGE_DIR  = "/root/qwen3-vl/Qwen3-VL-Embedding-2B/images"
EMBED_API  = "http://127.0.0.1:8848/v1/embeddings"

# 这里的大模型是指经过 `train_qwen_lora.py` 结构化属性抽取微调后的那套模型
LLM_API_KEY  = os.getenv("LLM_API_KEY", "EMPTY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8000/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

client = MilvusClient(DB_PATH)
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)


def parse_query_structured(query: str) -> dict:
    """
    步骤 1: 将原始 Query 喂给微调后的 LLM。
    其已被 SFT 定向训练，保证会按照 JSON 格式吐出:
    { "rewritten": "...", "attributes": { "category": "...", "color": "..." } }
    """
    system_prompt = (
        "你是一个电商搜索意图理解专家。请同时完成以下两项任务：\n"
        "1. 对用户的搜索词进行泛化与改写，生成更详细的搜索短语。\n"
        "2. 从用户的搜索词中提取具体商品属性（category，color）。未提及则留空。\n"
        "请严格只输出合法JSON，且键必须为 rewritten 和 attributes。"
    )
    
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.1,  # 确保 JSON 输出格式稳定
        max_tokens=256
    )
    
    raw_content = response.choices[0].message.content.strip()
    
    try:
        # 尝试清洗格式化并由于模型可能带 Markdown 标记 ```json ... ```，通常需要剥离它
        if raw_content.startswith('```json'):
            raw_content = raw_content[7:-3].strip()
            
        result = json.loads(raw_content)
        return result
    except json.JSONDecodeError as e:
        print(f"⚠️ 意图解析 JSON 失败, 回退降级。原因: {e}")
        # 强行返回兜底
        return {
            "rewritten": query, 
            "attributes": {"category": "", "color": ""}
        }

def embed_text(text: str):
    """请求 Qwen3-VL-Embedding 获取改写后文本的特征 (Dense Retriever)"""
    resp = session.post(EMBED_API, json={"text": text}, timeout=30)
    return resp.json()["data"]

def filter_ann_search(parsed_data: dict, top_k=3):
    """
    步骤 2: Filter-ANN (双流融合召回)
    根据 LLM 提取出的 Category 与 Color 属性拼装 Milvus Boolean Filter。
    并将 Rewritten 向量发往 Milvus 做 HNSW 空间检索。
    """
    rewritten_query = parsed_data.get("rewritten", "")
    attributes = parsed_data.get("attributes", {})
    
    # 提取特征
    vec = embed_text(rewritten_query)
    
    # 组装 Expr (Boolean Expression)
    filters = []
    category = attributes.get("category", "")
    color = attributes.get("color", "")
    
    if category:
        # 使用 LIKE 或者是双等号精确匹配视具体存储值而定
        filters.append(f"category == '{category}'")
    if color:
        filters.append(f"color == '{color}'")
        
    expr = " and ".join(filters) if filters else ""
    
    print(f"  [引擎端] 触发条件: expr='{expr}'")
    
    search_params = {
        "collection_name": COLLECTION,
        "data": [vec],
        "limit": top_k,
        "output_fields": ["filename", "category", "color"]
    }
    
    if expr:
        search_params["filter"] = expr
        
    try:
         results = client.search(**search_params)
    except Exception as e:
         print(f"  [引擎端] Filter 执行产生拦截或失败: {e}，回退至全局召回。")
         search_params.pop("filter")
         results = client.search(**search_params)
         
    return results[0]

def main():
    print("====================================")
    print("   Filter-ANN 增强多模态 RAG (V2)")
    print("====================================")

    while True:
        query = input("\n请输入搜索需求 (输入 q 退出): ").strip()
        if not query:
             continue
        if query.lower() == 'q':
             break
             
        # 1. 意图重组
        print(f"⏳ [1/2] 大模型正在拆解与泛化搜索意图...")
        parsed_intent = parse_query_structured(query)
        print(f"   改写文本: {parsed_intent['rewritten']}")
        print(f"   提取属性: {json.dumps(parsed_intent['attributes'], ensure_ascii=False)}")
        
        # 2. Filter-ANN 双流召回
        print(f"\n⏳ [2/2] Milvus 正在执行标量过滤 + 稠密向量搜索...")
        results = filter_ann_search(parsed_intent, top_k=3)
        
        if not results:
             print("❌ 未检索到符合这些严格条件的商品图。")
             continue
             
        print(f"✅ 成功命中 {len(results)} 件最优商品：")
        for i, r in enumerate(results):
            entity = r['entity']
            fname = entity['filename']
            cat = entity.get('category', '未知')
            col = entity.get('color', '未知')
            score = r['distance']
            print(f"   [{i+1}] {fname} | Label:[{col} {cat}] | 空间距离:{score:.4f}")

if __name__ == '__main__':
    main()
