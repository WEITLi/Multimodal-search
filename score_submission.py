"""
score_submission.py
将 predict_kdd.py 生成的 submission.csv 与 answer JSON 做离线评测。
指标与 eval_kdd.py 完全一致：NDCG@5、Recall@1、Recall@5。

用法：
    python score_submission.py \
        --submission test/submission.csv \
        --answer    data/ans/ans/testA_answer.json
"""

import json
import argparse
import numpy as np
import pandas as pd


# ── 指标函数（与 eval_kdd.py 保持完全一致）────────────────────────────────────

def compute_ndcg(retrieved_ids, ground_truth_ids, k=5):
    rel = [1 if pid in ground_truth_ids else 0 for pid in retrieved_ids[:k]]
    if not any(rel):
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth_ids), k)))
    if idcg == 0.0:
        return 0.0
    dcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
    return dcg / idcg


def compute_recall(retrieved_ids, ground_truth_ids, k=5):
    hits = sum(1 for pid in retrieved_ids[:k] if pid in ground_truth_ids)
    return hits / len(ground_truth_ids) if ground_truth_ids else 0.0


# ── 主评测流程 ────────────────────────────────────────────────────────────────

def score(submission_path: str, answer_path: str):
    # 读取答案（key 为字符串 query_id，value 为 int 列表）
    with open(answer_path, encoding="utf-8") as f:
        answers = json.load(f)
    # 统一转为字符串 set，方便与 submission 做比对
    answers = {str(k): set(str(v) for v in vs) for k, vs in answers.items()}

    # 读取提交文件
    sub = pd.read_csv(submission_path, dtype=str)
    # 期望列：query-id, product1 … product5
    prod_cols = [c for c in sub.columns if c.startswith("product")]

    ndcg_scores   = []
    recall1_scores = []
    recall5_scores = []
    missing        = []

    for _, row in sub.iterrows():
        qid = str(row["query-id"])
        if qid not in answers:
            missing.append(qid)
            continue

        retrieved   = [str(row[c]) for c in prod_cols]
        ground_truth = answers[qid]

        ndcg_scores.append(compute_ndcg(retrieved, ground_truth, k=5))
        recall1_scores.append(compute_recall(retrieved, ground_truth, k=1))
        recall5_scores.append(compute_recall(retrieved, ground_truth, k=5))

    n = len(ndcg_scores)
    print(f"{'─'*45}")
    print(f"  Evaluated queries : {n}")
    if missing:
        print(f"  ⚠️  Queries not in answer file: {len(missing)}")
    print(f"{'─'*45}")
    print(f"  NDCG@5    : {np.mean(ndcg_scores):.4f}")
    print(f"  Recall@1  : {np.mean(recall1_scores):.4f}")
    print(f"  Recall@5  : {np.mean(recall5_scores):.4f}")
    print(f"{'─'*45}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="test/submission.csv")
    parser.add_argument("--answer",     default="data/ans/ans/testA_answer.json")
    args = parser.parse_args()

    score(args.submission, args.answer)
