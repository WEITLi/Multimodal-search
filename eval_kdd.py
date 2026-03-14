import os
import json
import base64
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from transformers import AutoTokenizer

# 导入训练脚本中的数据解码及模型定义
from train_kdd_multimodal import KDDMultimodalDataset, VisualProjector

def compute_ndcg(retrieved_ids, ground_truth_ids, k=5):
    """
    计算单个 Query 的 NDCG@k
    按照 KDD 评估要求，验证集 Ground Truth 是一组等权重的无序集合
    """
    rel = [1 if pid in ground_truth_ids else 0 for pid in retrieved_ids[:k]]
    if not any(rel):
        return 0.0

    # 理想的最大 DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth_ids), k)))
    if idcg == 0.0:
        return 0.0
        
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
    return dcg / idcg

def compute_recall(retrieved_ids, ground_truth_ids, k=5):
    """计算 Recall@k"""
    hits = sum(1 for pid in retrieved_ids[:k] if pid in ground_truth_ids)
    return hits / len(ground_truth_ids) if ground_truth_ids else 0.0

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation on device: {device}")

    # ===== 1. 加载模型 =====
    MODEL_PATH = "/root/qwen3-vl/Qwen3-VL-Embedding-2B"
    try:
        from transformers import AutoModel
        text_encoder = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Mocking models for dry-run... {e}")
        text_encoder = nn.Linear(64, 4096).to(device)
        tokenizer = type('MockTokenizer', (object,), {
            "__call__": lambda self, t, **kwargs: {'input_ids': torch.zeros((1, 64), dtype=torch.long), 'attention_mask': torch.ones((1, 64), dtype=torch.long)}
        })()

    visual_projector = VisualProjector(out_dim=4096).to(device)
    
    weights_path = "kdd_visual_projector_qwen3_2B.pth"
    if os.path.exists(weights_path):
        visual_projector.load_state_dict(torch.load(weights_path, map_location=device))
        print("✅ Loaded trained Visual Projector weights.")
    else:
        print("⚠️ Custom projector weights not found, using randomly initialized weights!")

    text_encoder.eval()
    visual_projector.eval()

    # ===== 2. 读取 Valid 数据与 Answer =====
    VALID_TSV = "multimodal_valid/valid.tsv"
    ANSWER_JSON = "multimodal_valid/valid_answer.json"

    if not os.path.exists(VALID_TSV) or not os.path.exists(ANSWER_JSON):
        print(f"Dataset missing at {VALID_TSV} or {ANSWER_JSON}. Exiting dry run.")
        return

    print("Loading valid answers...")
    with open(ANSWER_JSON, 'r') as f:
        ground_truth = json.load(f)

    print("Loading valid TSV...")
    dataset = KDDMultimodalDataset(VALID_TSV, tokenizer)
    
    # 将候选按 Query-ID 分组
    queries = defaultdict(list)
    for idx in range(len(dataset.df)):
        row = dataset.df.iloc[idx]
        qid = str(row['query_id'])
        queries[qid].append((idx, str(row['product_id'])))

    # ===== 3. 执行评估 =====
    ndcg_scores = []
    recall1_scores = []
    recall5_scores = []

    print(f"Starting evaluation on {len(queries)} queries...")
    with torch.no_grad():
        for i, (qid, candidates) in enumerate(queries.items()):
            # 同一个 qid 下的第一条数据的 input_ids 都是一样的
            first_idx = candidates[0][0]
            data = dataset[first_idx]
            
            # 文本向量提取
            t_input = data['input_ids'].unsqueeze(0).to(device) # [1, 64]
            t_feat = text_encoder(t_input.float()) if isinstance(text_encoder, nn.Linear) else text_encoder(t_input)
            t_feat = torch.nn.functional.normalize(t_feat, dim=-1) # [1, 4096]
            
            # 候选图片的视觉特征提取
            cand_pids = []
            cand_feats = []
            
            for idx, pid in candidates:
                c_data = dataset[idx]
                r_feat = c_data['roi_features'].unsqueeze(0).to(device)
                r_box = c_data['roi_boxes'].unsqueeze(0).to(device)
                r_cls = c_data['roi_classes'].unsqueeze(0).to(device)
                r_mask = c_data['roi_mask'].unsqueeze(0).to(device)
                
                v_feat = visual_projector(r_feat, r_box, r_cls, r_mask)
                v_feat = torch.nn.functional.normalize(v_feat, dim=-1)
                
                cand_feats.append(v_feat)
                cand_pids.append(pid)
                
            cand_feats = torch.cat(cand_feats, dim=0) # [Num_Cand, 4096]
            
            # 计算 Cosine Similarity 
            # [1, 4096] @ [4096, Num_Cand] -> [1, Num_Cand]
            sims = torch.matmul(t_feat, cand_feats.T).squeeze(0)
            
            # 排序获取 Top-K
            sorted_indices = torch.argsort(sims, descending=True).cpu().numpy()
            retrieved_pids = [cand_pids[idx] for idx in sorted_indices]
            
            gt_pids = ground_truth.get(qid, [])
            
            # 计算指标
            ndcg_scores.append(compute_ndcg(retrieved_pids, gt_pids, k=5))
            recall1_scores.append(compute_recall(retrieved_pids, gt_pids, k=1))
            recall5_scores.append(compute_recall(retrieved_pids, gt_pids, k=5))

            if (i+1) % 50 == 0:
                print(f"[{i+1}/{len(queries)}] Running NDCG@5: {np.mean(ndcg_scores):.4f}")

    print("\n========= Final Valid Evaluation =========")
    print(f"Total Queries: {len(queries)}")
    print(f"Recall@1: {np.mean(recall1_scores):.4f}")
    print(f"Recall@5: {np.mean(recall5_scores):.4f}")
    print(f"NDCG@5:   {np.mean(ndcg_scores):.4f}")
    print("==========================================")

if __name__ == "__main__":
    evaluate()
