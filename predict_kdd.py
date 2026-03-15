import os
import csv
import torch
import torch.nn as nn
from collections import defaultdict
from transformers import AutoTokenizer

from train_kdd_multimodal import KDDMultimodalDataset, VisualProjector

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generation on device: {device}")

    # ===== 1. 载入模型 =====
    MODEL_PATH = ".pretrain_models/Qwen3-VL-Embedding-2B"
    try:
        from transformers import AutoModel
        text_encoder = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except:
        print("Mocking models for dry-run...")
        text_encoder = nn.Linear(64, 2048).to(device)
        tokenizer = type('Mock', (object,), {
            "__call__": lambda self, t, **k: {'input_ids': torch.zeros((1, 64), dtype=torch.long), 'attention_mask': torch.ones((1, 64), dtype=torch.long)}
        })()

    visual_projector = VisualProjector(out_dim=2048).to(device)
    weights_path = "sft/kdd_visual_projector_qwen3_2B.pth"
    if os.path.exists(weights_path):
        visual_projector.load_state_dict(torch.load(weights_path, map_location=device))
        print("✅ Loaded trained Visual Projector weights.")

    text_encoder.eval()
    visual_projector.eval()

    # ===== 2. 读取测试集 (这里配置需要生成的集，比如 testA) =====
    TEST_TSV = "data/multimodal_testA/testA.tsv"   # 按需替换 testB.tsv
    OUTPUT_CSV = "test/submission.csv"

    if not os.path.exists(TEST_TSV):
        print(f"Dataset missing at {TEST_TSV}")
        return

    print("Loading test TSV...")
    dataset = KDDMultimodalDataset(TEST_TSV, tokenizer)
    
    queries = defaultdict(list)
    for idx in range(len(dataset.df)):
        row = dataset.df.iloc[idx]
        qid = str(row['query_id'])
        queries[qid].append((idx, str(row['product_id'])))

    # ===== 3. 生成预测并写入 CSV =====
    print(f"Generating Top-5 predictions for {len(queries)} queries...")
    
    with open(OUTPUT_CSV, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # 写表头
        writer.writerow(['query-id', 'product1', 'product2', 'product3', 'product4', 'product5'])
        
        with torch.no_grad():
            for i, (qid, candidates) in enumerate(queries.items()):
                first_idx = candidates[0][0]
                data = dataset[first_idx]
                
                t_input = data['input_ids'].unsqueeze(0).to(device)
                t_mask  = data['attention_mask'].unsqueeze(0).to(device)
                if isinstance(text_encoder, nn.Linear):
                    t_feat = text_encoder(t_input.float())
                else:
                    out    = text_encoder(input_ids=t_input, attention_mask=t_mask)
                    hidden = out.last_hidden_state
                    mask_e = t_mask.unsqueeze(-1).expand(hidden.size()).float()
                    t_feat = (hidden * mask_e).sum(1) / mask_e.sum(1).clamp(min=1e-9)
                t_feat = torch.nn.functional.normalize(t_feat, dim=-1)
                
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
                    
                cand_feats = torch.cat(cand_feats, dim=0)
                sims = torch.matmul(t_feat, cand_feats.T).squeeze(0)
                
                sorted_indices = torch.argsort(sims, descending=True).cpu().numpy()
                retrieved_pids = [cand_pids[idx] for idx in sorted_indices[:5]]
                
                # 若不足5个商品候选，用最后一位补齐 (实际上KDD一般有30个)
                while len(retrieved_pids) < 5:
                    retrieved_pids.append(retrieved_pids[-1] if retrieved_pids else "0")
                
                # 写入 [query-id, prod1, prod2...]
                writer.writerow([qid] + retrieved_pids)

                if (i+1) % 100 == 0:
                    print(f"Processed [{i+1}/{len(queries)}]")

    print(f"✅ Prediction completed. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    predict()
