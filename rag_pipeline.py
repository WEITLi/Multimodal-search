import os
import requests
import base64
from pymilvus import MilvusClient
from openai import OpenAI

# 绕过系统代理
session = requests.Session()
session.trust_env = False

# 配置路径与服务地址
DB_PATH    = "/root/qwen3-vl/Qwen3-VL-Embedding-2B/milvus_qwen3vl.db"
IMAGE_DIR  = "/root/qwen3-vl/Qwen3-VL-Embedding-2B/images"
COLLECTION = "qwen3_vl_images"
EMBED_API  = "http://127.0.0.1:8848/v1/embeddings"

# 大模型 API 配置 (支持 vLLM / DashScope / Ollama 等兼容 OpenAI 的接口)
# 默认指向本地 8000 端口，模型名称按实际启动的服务填写
LLM_API_KEY  = os.getenv("LLM_API_KEY", "EMPTY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8000/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

client = MilvusClient(DB_PATH)
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)


def embed_text(text: str):
    """请求 Qwen3-VL-Embedding 获取文本特征"""
    resp = session.post(EMBED_API, json={"text": text}, timeout=30)
    return resp.json()["data"]


def multimodal_search(query: str, top_k=3):
    """在 Milvus 中执行向量检索"""
    vec = embed_text(query)
    results = client.search(
        collection_name=COLLECTION,
        data=[vec],
        limit=top_k,
        output_fields=["filename"]
    )
    return results[0]


def rewrite_query(query: str) -> str:
    """步骤 1: LLM 意图泛化与改写"""
    prompt = (
        "你是一个电商搜索意图理解专家。请对用户的搜索词进行泛化与改写，"
        "提取其中的核心实体与特征属性（比如：材质、款式、颜色、风格、使用场景等），"
        "并生成一段结构清晰且语义丰富的搜索短语，以提升多模态向量检索的召回率。\n"
        "请直接输出改写后的文本，不要带有任何多余解释词汇。\n\n"
        f"用户原始输入: {query}\n"
        "改写结果:"
    )
    
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, # 较低的温度保证提取事实的准确性
        max_tokens=128
    )
    return response.choices[0].message.content.strip()


def encode_image(image_path: str) -> str:
    """Base64 编码图片"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_recommendation(original_query: str, search_results: list) -> str:
    """步骤 3: VLM 融合检索到的商品图进行推荐话术生成"""
    messages = [
        {
            "role": "system", 
            "content": "你是一个金牌电商导购员。请根据用户的查询意图以及系统检索出的一组商品图片，生成一段热情、专业的推荐话术。请指出这些商品的外观亮点并说明它们为何符合用户的预期。"
        }
    ]
    
    # 构造多模态 prompt
    content = [
        {
            "type": "text", 
            "text": f"用户的搜索需求是: {original_query}\n\n以下是我们数据库中检索到的关联度最高的商品图片，请基于图片信息向用户进行推介："
        }
    ]
    
    for i, item in enumerate(search_results):
        fname = item["entity"]["filename"]
        img_path = os.path.join(IMAGE_DIR, fname)
        if os.path.exists(img_path):
            base64_img = encode_image(img_path)
            content.append({"type": "text", "text": f"\n商品 {i+1} ({fname}):\n"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })
            
    messages.append({"role": "user", "content": content})
    
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,  # 适中的温度保持回复的生动性
        max_tokens=600
    )
    return response.choices[0].message.content


def main():
    print("====================================")
    print("      多模态 RAG 检索生成系统")
    print("====================================")
    print(f"向量库路径: {DB_PATH}")
    print(f"大语言模型: {LLM_MODEL} @ {LLM_API_BASE}")
    print("------------------------------------\n")

    while True:
        query = input("请输入搜索需求 (输入 q 退出): ").strip()
        if not query:
            continue
        if query.lower() == 'q':
            break
            
        print("\n⏳ [1/3] LLM 正在分析并改写查询意图...")
        try:
            rewritten_query = rewrite_query(query)
            print(f"✅ 改写后 Query: {rewritten_query}")
        except Exception as e:
            print(f"❌ 意图分析失败，回退至原始 Query: {e}")
            rewritten_query = query
            
        print("\n⏳ [2/3] Milvus 正在进行跨模态向量检索...")
        try:
            results = multimodal_search(rewritten_query, top_k=3)
            print(f"✅ 成功找到 {len(results)} 件候选商品：")
            for i, r in enumerate(results):
                print(f"   [{i+1}] {r['entity']['filename']} (距离/相似度: {r['distance']:.4f})")
        except Exception as e:
            print(f"❌ 检索失败: {e}\n")
            continue
            
        print("\n⏳ [3/3] VLM 大模型正在根据商品实拍生成推荐话术...")
        try:
            response = generate_recommendation(query, results)
            print("\n================  RAG 生成推荐结果  ================\n")
            print(response)
            print("\n====================================================\n")
        except Exception as e:
            print(f"❌ 生成推荐话术失败: {e}\n")


if __name__ == '__main__':
    main()
