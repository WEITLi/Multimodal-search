import os
import json
import base64
import requests
from pymilvus import MilvusClient, DataType

session = requests.Session()
session.trust_env = False

DB_PATH = "./pretrain_models/Qwen3-VL-Embedding-2B/milvus_v2_filter_ann.db"
client = MilvusClient(DB_PATH)
collection_name = "qwen3_vl_images_with_attributes"
IMAGE_DIR = "./pretrain_models/Qwen3-VL-Embedding-2B/images"
ATTRIBUTES_DB_PATH = "sft/item_attributes.json"

def create_collection_with_subgraphs():
    """
    根据给定的聚类特征与属性特征，构建特殊的索引结构:
    (1) 向量聚类子图构建（Milvus底层的 IVF / HNSW 聚类分簇）
    (2) 属性导航点子图构建 (倒排 + 标量过滤)
    最后引擎端会进行 Reduce 合并
    """
    if collection_name in client.list_collections():
        print("Collection (V2) 已存在，继续追加入库")
        return
        
    print("Creating Schema for Multi-Subgraph Filter-ANN Indexing...")
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    
    # Primary Key
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=500)
    
    # Dense Vector: 4096 维图像特征，用于构建「向量聚类子图」
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=4096)
    
    # Metadata Fields: 这里保存属性，用于构建「属性子图」的导航点
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=200, default_value="")
    schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=100, default_value="")

    schema.verify()
    
    # --- Indexing Strategy (索引合并概念) ---
    index_params = client.prepare_index_params()
    
    # 1. 向量聚类子图索引 (在这里我们用 HNSW 模拟导航点遍历)
    index_params.add_index(
        field_name="vector", 
        index_type="HNSW", # 使用基于图的 HNSW 算法体现“路径遍历检索点”
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200} 
    )
    
    # 2. 属性子图索引 (倒排)，构建离散的属性导航点
    # 在 Milvus 底层这体现为标量的 Trie 或 Bitmap 索引，后续和向量图取交集
    index_params.add_index(field_name="category", index_type="Trie")
    index_params.add_index(field_name="color", index_type="Trie")
    
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    print("✅ Multi-Subgraph Collection 创建成功！合并策略准备就绪。")

def load_attributes():
    if not os.path.exists(ATTRIBUTES_DB_PATH):
        print(f"⚠️ {ATTRIBUTES_DB_PATH} not found. Ensure `extract_visual_properties.py` is run first.")
        return {}
    with open(ATTRIBUTES_DB_PATH, 'r') as f:
        return json.load(f)

def embed_image(image_path):
    with open(image_path, "rb") as f:
         b64 = base64.b64encode(f.read()).decode('utf-8')
    response = session.post(
        "http://127.0.0.1:8848/v1/embeddings",
        json={"image": b64},
        timeout=60
    )
    return response.json()["data"]

def main():
    create_collection_with_subgraphs()
    attributes_db = load_attributes()
    
    all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(all_images)
    print(f"待处理: {total} 张")

    batch = []
    for i, img_name in enumerate(all_images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        pid = img_name.split('.')[0]
        item_attrs = attributes_db.get(pid, {"category": "", "color": ""})
        
        try:
            vec = embed_image(img_path)
            batch.append({
                "id": img_name, 
                "vector": vec, 
                "filename": img_name,
                "category": item_attrs.get("category", ""),
                "color": item_attrs.get("color", "")
            })
            if len(batch) >= 10:
                client.upsert(collection_name=collection_name, data=batch)
                batch = []
            if (i + 1) % 50 == 0:
                print(f"进度: {i+1}/{total}")
        except Exception as e:
             pass

    if batch:
        client.upsert(collection_name=collection_name, data=batch)
    print(f"✅ 入库完成，不同维度的子图导航节点已被写入。")

if __name__ == '__main__':
    print("Script ready for Filter-ANN multi-subgraph insertion.")
