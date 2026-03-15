import os
import json
import base64
import numpy as np
import pandas as pd


class VisualAttributeExtractor:
    """
    无原图模式下的商品属性提取器。
    输入：TSV 中预提取的 boxes（坐标）、class_labels（类别 id）、features（2048维 ResNet）
    输出：category（来自 multimodal_labels.txt 的文字）、color（无原图时标 unknown）

    如未来有原图，可扩展 extract_color_from_image() 分支。
    """

    # 来源：data/multimodal_labels.txt（33 类，id 0-32）
    CATEGORY_MAP = {
        0:  "top clothes (coat, jacket, shirt, etc.)",
        1:  "skirt & dress",
        2:  "bottom clothes (trousers, pants, etc.)",
        3:  "luggage, leather goods",
        4:  "shoes",
        5:  "accessories (jewelry, clothing accessories, belts, hats, scarves, etc.)",
        6:  "snacks, nuts, liquor and tea",
        7:  "makeup, perfume, beauty tools and essential oils",
        8:  "bottle drink",
        9:  "furniture",
        10: "stationery",
        11: "household electrical appliances",
        12: "home decoration",
        13: "household fabric",
        14: "kitchenware",
        15: "home / personal cleaning tools",
        16: "storage supplies",
        17: "motorcycle, motorcycle accessories, vehicles, bicycle and riding equipment",
        18: "outdoor product",
        19: "lighting",
        20: "toys",
        21: "underwear",
        22: "digital supplies",
        23: "bed linens",
        24: "baby products",
        25: "personal care",
        26: "sporting goods",
        27: "clothes (accessories, baby clothing, etc.)",
        28: "others",
        29: "human face",
        30: "arm",
        31: "hair",
        32: "hand",
    }

    def _dominant_class(
        self,
        boxes: np.ndarray,      # [num_boxes, 4]  (x1, y1, x2, y2)
        class_ids: np.ndarray,  # [num_boxes]
        image_w: float,
        image_h: float,
    ) -> int:
        """
        选出面积最大的 Box 对应的 class_id 作为商品主类别。
        与 train_kdd_multimodal.py 中归一化坐标的语义一致：
          boxes 列顺序为 [x1, y1, x2, y2]（归一化前的原始像素坐标）
        """
        widths  = np.clip(boxes[:, 2] - boxes[:, 0], 0, image_w)
        heights = np.clip(boxes[:, 3] - boxes[:, 1], 0, image_h)
        areas   = widths * heights
        dominant_idx = int(np.argmax(areas))
        return int(class_ids[dominant_idx])

    def extract_attributes(
        self,
        boxes: np.ndarray,      # [num_boxes, 4]
        class_ids: np.ndarray,  # [num_boxes]
        features: np.ndarray,   # [num_boxes, 2048]  预留，供未来扩展
        image_w: float,
        image_h: float,
    ) -> dict:
        """
        无原图模式：
          - category: 面积最大 Box 的 class_id → CATEGORY_MAP 文字
          - color:    2048维 ResNet 特征为语义特征而非颜色特征，无法可靠提取，标 unknown
        """
        dominant_cls = self._dominant_class(boxes, class_ids, image_w, image_h)
        category = self.CATEGORY_MAP.get(dominant_cls, f"cat_id_{dominant_cls}")
        return {
            "category": category,
            "color": "unknown",   # 无原图时无法提取颜色
        }


def _decode_b64_arr(b64_str: str, dtype, shape) -> np.ndarray:
    """与 train_kdd_multimodal.py 中 _decode_b64_arr 保持一致。"""
    return np.frombuffer(base64.b64decode(b64_str), dtype=dtype).reshape(shape)


def process_dataset(tsv_path: str, output_json: str = "./sft/item_attributes.json"):
    """
    离线批量提取商品属性并落盘，后续打入 Milvus 作为 Filter 字段。
    TSV 列名参考 train_kdd_multimodal.py：
      product_id, image_h, image_w, num_boxes, boxes, features, class_labels, query, query_id
    """
    extractor = VisualAttributeExtractor()

    df = pd.read_csv(
        tsv_path, sep="\t", header=None,
        names=["product_id", "image_h", "image_w", "num_boxes",
               "boxes", "features", "class_labels", "query", "query_id"],
    )

    attributes_db = {}
    print(f"Starting attribute extraction for {len(df)} rows...")

    for idx, row in df.iterrows():
        pid = str(row["product_id"])
        if pid in attributes_db:
            continue

        num_boxes = int(row["num_boxes"])
        image_w   = float(row["image_w"])
        image_h   = float(row["image_h"])

        boxes     = _decode_b64_arr(row["boxes"],        np.float32, (num_boxes, 4))
        features  = _decode_b64_arr(row["features"],     np.float32, (num_boxes, 2048))
        class_ids = _decode_b64_arr(row["class_labels"], np.int64,   (num_boxes,))

        attributes_db[pid] = extractor.extract_attributes(
            boxes, class_ids, features, image_w, image_h
        )

        if idx % 1000 == 0:
            print(f"  Processed {idx} rows...")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(attributes_db, f, indent=2, ensure_ascii=False)

    print(f"✅ Product attributes saved to {output_json}  ({len(attributes_db)} items)")


if __name__ == "__main__":
    TSV_FILE = "./data/multimodal_train_sampleset/train.sample.tsv"
    if os.path.exists(TSV_FILE):
        process_dataset(TSV_FILE)
    else:
        print(f"File not found: {TSV_FILE}")
