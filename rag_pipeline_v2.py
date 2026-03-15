import os
import json
import base64
import requests
from pymilvus import MilvusClient
from openai import OpenAI
import time

# ======= 基础配置 =======
session = requests.Session()
session.trust_env = False

DB_PATH    = "/root/qwen3-vl/Qwen3-VL-Embedding-2B/milvus_v2_filter_ann.db"
COLLECTION = "qwen3_vl_images_with_attributes"
EMBED_API  = "http://127.0.0.1:8848/v1/embeddings"

LLM_API_KEY  = os.getenv("LLM_API_KEY", "EMPTY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8000/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

client = MilvusClient(DB_PATH)
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)


def parse_query_structured(query: str) -> dict:
    """利用 LLM 将 Query 拆分为检索向量词汇与确定的属性过滤项"""
    system_prompt = (
        "你是一个电商搜索意图理解专家。请同时完成以下两项任务：\n"
        "1. 对用户的搜索词进行泛化与改写，生成更详细的搜索短语。\n"
        "2. 从用户的搜索词中提取具体商品属性（category，color）。未提及则留空。\n"
        "请严格只输出合法JSON，且键必须为 rewritten 和 attributes。"
    )
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=256
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith('```json'):
            raw = raw[7:-3].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"⚠️ 解析失败，回退降级处理。")
        return {"rewritten": query, "attributes": {"category": "", "color": ""}}

def embed_text(text: str):
    """请求模型获取高维文本特征向量"""
    resp = session.post(EMBED_API, json={"text": text}, timeout=30)
    return resp.json()["data"]

def multi_branch_search(parsed_data: dict, top_k=5):
    """
    双分支召回策略演示:
    分支 1: 向量召回 (不带任何过滤，全图探索)
    分支 2: 属性子图过滤召回 (约束导航点边界条件提取)
    最后将两种方式查找到的结果聚合去重。
    """
    rewritten_query = parsed_data.get("rewritten", "")
    attributes = parsed_data.get("attributes", {})
    
    # 提取特征
    vec = embed_text(rewritten_query)
    
    # ----------------------------------------------------
    # 分支一：纯向量召回 (遍历原始向量子图候选簇点)
    # ----------------------------------------------------
    print(f"\n  [分支 1: Vector Navigation] 正在全图域寻找相近语义路径点...")
    vector_results = client.search(
        collection_name=COLLECTION,
        data=[vec],
        limit=top_k, # 设置召回数后停止
        output_fields=["filename", "category", "color"]
    )[0]
    
    # ----------------------------------------------------
    # 分支二：属性过滤召回 (以属性作为起始导航点，只遍历具有相同属性的候选点)
    # ----------------------------------------------------
    filters = []
    category = attributes.get("category", "")
    color = attributes.get("color", "")
    
    if category: filters.append(f"category == '{category}'")
    if color: filters.append(f"color == '{color}'")
    
    expr = " and ".join(filters) if filters else ""
    
    attribute_results = []
    if expr:
        print(f"  [分支 2: Attribute Navigation] 命中属性组 ({expr})，正在特定属性子图中遍历与打分...")
        try:
            attribute_results = client.search(
                collection_name=COLLECTION,
                data=[vec],
                filter=expr,
                limit=top_k, 
                output_fields=["filename", "category", "color"]
            )[0]
        except Exception as e:
            print(f"    属性分支异常撤解: {e}")
    else:
        print(f"  [分支 2: Attribute Navigation] 没有提取出强制属性，跳过属性过滤召回。")

    # ----------------------------------------------------
    # 合并聚类 (Reduce)
    # ----------------------------------------------------
    merged_results = {}
    
    # 优先将属性命中项纳入结果（因为硬条件必然符合用户要求，在业务中权值可以放高）
    for res in attribute_results:
        pid = res['entity']['filename']
        merged_results[pid] = {
            "source": "Attr-Filter",
            "distance": res['distance'],
            "entity": res['entity']
        }
        
    # 加入纯向量召回的补充池 (去重逻辑)
    for res in vector_results:
        pid = res['entity']['filename']
        if pid not in merged_results:
            merged_results[pid] = {
                "source": "Global-Vector",
                "distance": res['distance'], 
                "entity": res['entity']
            }
            
    # 从合并后的候选池中再按相似度排序截断给出最终输出
    final_list = list(merged_results.values())
    final_list.sort(key=lambda x: x['distance'], reverse=True) # COSINE 是越大越相似
    
    return final_list[:top_k]

def main():
    print("=====================================================")
    print("  Custom Filter-ANN Multi-Branch RAG Pipeline")
    print("=====================================================")

    while True:
        query = input("\n请输入搜索需求 (输入 q 退出): ").strip()
        if not query: continue
        if query.lower() == 'q': break
             
        # 1. 意图解析
        parsed_intent = parse_query_structured(query)
        print(f"   [LLM 提取]: Rewritten='{parsed_intent['rewritten']}', Attrs={json.dumps(parsed_intent['attributes'], ensure_ascii=False)}")
        
        # 2. 双流召回与合并
        t0 = time.time()
        results = multi_branch_search(parsed_intent, top_k=5)
        print(f"   ➤ 多路召回引擎耗时: {time.time()-t0:.3f} 秒")
        
        if not results:
             print("❌ 未检索到相关商品。")
             continue
             
        print(f"✅ 最终合成 Top-{len(results)} 商品打分：")
        for i, r in enumerate(results):
            entity = r['entity']
            fname = entity['filename']
            cat = entity.get('category', '未知')
            col = entity.get('color', '未知')
            source = r['source']
            score = r['distance']
            print(f"   [{i+1}] {fname} | Label:[{col} {cat}] | 空间距离:{score:.4f} | 召回来源:[{source}]")

if __name__ == '__main__':
    main()
