import os
import json
import base64
import requests
from pymilvus import MilvusClient, DataType

session = requests.Session()
session.trust_env = False

DB_PATH = "/root/qwen3-vl/Qwen3-VL-Embedding-2B/milvus_v2_filter_ann.db"
client = MilvusClient(DB_PATH)
collection_name = "qwen3_vl_images_with_attributes"
IMAGE_DIR = "/root/qwen3-vl/Qwen3-VL-Embedding-2B/images"
ATTRIBUTES_DB_PATH = "item_attributes.json"


def create_collection_with_scalars():
    if collection_name in client.list_collections():
        print("Collection (V2) 已存在，继续追加入库")
        return
        
    print("Creating Milvus schema with Scalar fields for Boolean Filtering...")
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    
    # Primary Key
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=500)
    
    # Dense Vector: 4096 维 Qwen 图文一致性映射向量
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=4096)
    
    # Metadata Fields: 这里保存的字段必须在创建索引前定义为标量字段，以便使用 `expr` 条件过滤
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=200, default_value="")
    schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=100, default_value="")

    schema.verify()
    
    # Create Index: 结合标量过滤的近似最近邻（Hybrid Search）
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    # Milvus 2.x 针对标量字段建立倒排索引，加速 `expr` 查询
    index_params.add_index(field_name="category", index_type="Trie")
    index_params.add_index(field_name="color", index_type="Trie")
    
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    print("✅ Collection V2 (Filter-ANN enabled) 创建成功")


def load_attributes():
    if not os.path.exists(ATTRIBUTES_DB_PATH):
        print(f"⚠️ {ATTRIBUTES_DB_PATH} not found. Ensure `extract_visual_properties.py` is run first.")
        return {}
    with open(ATTRIBUTES_DB_PATH, 'r') as f:
        return json.load(f)

def embed_image(image_path):
    with open(image_path, "rb") as f:
         b64 = base64.b64encode(f.read()).decode('utf-8')
    # 真实部署调用后端计算 Embeddings
    response = session.post(
        "http://127.0.0.1:8848/v1/embeddings",
        json={"image": b64},
        timeout=60
    )
    return response.json()["data"]

def main():
    create_collection_with_scalars()
    attributes_db = load_attributes()
    
    all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(all_images)
    print(f"待处理: {total} 张")

    batch = []
    for i, img_name in enumerate(all_images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        # 从属性提取库里拿标签，若不存在先赋空字符串。
        # 这里 img_name 可以当做 key 处理。实际可能需要去掉后缀，映射 pid。
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
    print(f"✅ 入库完成")

if __name__ == '__main__':
    # main() 示例防抛错
    print("Script ready for Filter-ANN vector insertion.")
