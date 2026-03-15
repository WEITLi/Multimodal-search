import os
import base64
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# 假设这里引入你封装好的模型架构
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
# 因为你需要纯写代码而暂时不激活环境，我们Mock一个导入或者直接手写组件调用

class KDDMultimodalDataset(Dataset):
    """
    处理 KDD Cup 2020 淘宝多模态搜索数据集的 Dataset 
    格式：9 列表格 TSV 
    """
    def __init__(self, tsv_file, tokenizer, max_boxes=36, max_text_len=64):
        print(f"Loading dataset from {tsv_file}...")
        self.df = pd.read_csv(tsv_file, sep='\t', header=0,
                              names=['product_id', 'image_h', 'image_w', 'num_boxes',
                                     'boxes', 'features', 'class_labels', 'query', 'query_id'])
        self.tokenizer = tokenizer
        self.max_boxes = max_boxes
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.df)

    def _decode_b64_arr(self, b64_str, dtype, shape):
        b = base64.b64decode(b64_str)
        return np.frombuffer(b, dtype=dtype).reshape(shape)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. 解析文本 Query
        text_query = str(row['query'])
        # Qwen3-VL 的 tokenizer 解析
        text_inputs = self.tokenizer(
            text_query, 
            max_length=self.max_text_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )

        # 2. 解析图像预打分 Box Features (2048维) + Box坐标 + 类别
        num_boxes = int(row['num_boxes'])
        w, h = float(row['image_w']), float(row['image_h'])
        
        # 限制最大 box 数量
        valid_boxes = min(num_boxes, self.max_boxes)
        
        b64_feat = row['features']
        b64_box = row['boxes']
        b64_cls = row['class_labels']

        # 恢复numpy数组
        features = self._decode_b64_arr(b64_feat, np.float32, (num_boxes, 2048))[:valid_boxes]
        boxes = self._decode_b64_arr(b64_box, np.float32, (num_boxes, 4))[:valid_boxes]
        class_ids = self._decode_b64_arr(b64_cls, np.int64, (num_boxes,))[:valid_boxes]

        # 组装完整的视觉特征：2048(ResNet) + 4(归一化坐标) + 2(面积/宽高比) 
        # (遵循 KDD WinnieTheBest 方案中的 Box 增强)
        norm_boxes = np.zeros((valid_boxes, 6), dtype=np.float32)
        norm_boxes[:, 0] = boxes[:, 0] / w  # x1
        norm_boxes[:, 1] = boxes[:, 1] / h  # y1
        norm_boxes[:, 2] = boxes[:, 2] / w  # x2
        norm_boxes[:, 3] = boxes[:, 3] / h  # y2
        norm_boxes[:, 4] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (w * h) # area
        norm_boxes[:, 5] = (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1] + 1e-6) # aspect ratio
        
        # Pad 到 max_boxes
        pad_len = self.max_boxes - valid_boxes
        attention_mask = np.zeros(self.max_boxes, dtype=np.int64)
        attention_mask[:valid_boxes] = 1
        
        if pad_len > 0:
            features = np.pad(features, ((0, pad_len), (0, 0)), 'constant')
            norm_boxes = np.pad(norm_boxes, ((0, pad_len), (0, 0)), 'constant')
            class_ids = np.pad(class_ids, (0, pad_len), 'constant')

        return {
            "input_ids": text_inputs['input_ids'].squeeze(0),
            "attention_mask": text_inputs['attention_mask'].squeeze(0),
            "roi_features": torch.tensor(features),           # [max_boxes, 2048]
            "roi_boxes": torch.tensor(norm_boxes),            # [max_boxes, 6]
            "roi_classes": torch.tensor(class_ids),           # [max_boxes]
            "roi_mask": torch.tensor(attention_mask)          # [max_boxes]
        }


class VisualProjector(nn.Module):
    """
    考虑到 KDD 数据集只有提取好的 2048维特征，无法直接喂给 Qwen3-VL的Vision Encoder，
    这里构建一个视觉投影融合网络：将离散的 Box Features (2048) 融合投影成 4096 维的全局向量，
    以对应 Qwen3-VL 文本端输出的 4096 维联合空间。
    """
    def __init__(self, in_features=2048, box_dim=6, num_classes=34, class_embed_dim=32, out_dim=4096, hidden_dim=1024):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        
        # 融合维度: 2048 + 6 + 32 = 2086
        merged_dim = in_features + box_dim + class_embed_dim
        
        # Self-Attention 进行实体间的语义交互
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=merged_dim, nhead=14, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        # 将聚合后的特征映射到 Qwen3-VL 的空间 (4096D)
        self.fc_proj = nn.Sequential(
            nn.Linear(merged_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, out_dim)
        )

    def forward(self, roi_feat, roi_box, roi_cls, roi_mask):
        cls_emb = self.class_embedding(roi_cls)
        # [batch, seq_len, 2086]
        x = torch.cat([roi_feat, roi_box, cls_emb], dim=-1)
        
        # Transformer (注意 PyTorch 的 key_padding_mask 中 True 代表需要 Mask)
        padding_mask = (roi_mask == 0)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Mean Pooling 分支 (忽略 padding 的 zero 分数)
        mask_expanded = roi_mask.unsqueeze(-1).expand_as(x).float()
        x_pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        # 映射到 4096 维度
        return self.fc_proj(x_pooled)


class ITCA_Loss(nn.Module):
    """
    Image-Text Contrastive Alignment (ITCA) Loss
    参考 CLIP 训练方法：对比批次内的正样本图文对与负样本图文对
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, image_features, text_features):
        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        # matrix对角线上是正样本,非对角线是负样本
        logit_scale = self.logit_scale.exp()
        # 计算相似度
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        batch_size = image_features.shape[0]
        # 从0 -> N 的index,对于第i个样本，对应的正确文本在i_th col
        labels = torch.arange(batch_size, device=image_features.device)
        # 对称交叉熵: 让对角线上的数值最大,非对角线最小
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==========================
    # 1. 模型及 Tokenizer 初始化
    # ==========================
    MODEL_PATH = "./pretrain_models/Qwen3-VL-Embedding-2B"
    try:
        from transformers import AutoModel
        # Qwen3-VL 的文本编码部分（实际部署时可接入含 LoRA 的 PEFT）
        text_encoder = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"加载预训练模型失败或环境未激活 (当前演示模式): {e}")
        # 在只写代码且无环境要求下，以占位符略过
        tokenizer = type('MockTokenizer', (object,), {
            "__call__": lambda self, t, **kwargs: {'input_ids': torch.zeros((1, 64), dtype=torch.long), 'attention_mask': torch.ones((1, 64), dtype=torch.long)}
        })()
        text_encoder = nn.Linear(64, 4096).to(device) # Mock 

    # 冻结文本大模型的主干（或仅开启 LoRA 微调）
    for param in text_encoder.parameters():
        param.requires_grad = False
        
    # Qwen3-VL-2B 的 hidden_size = 2048，视觉侧输出维度必须与之对齐
    visual_projector = VisualProjector(out_dim=2048).to(device)
    criterion = ITCA_Loss().to(device)

    # ==========================
    # 2. 数据加载
    # ==========================
    train_dataset = KDDMultimodalDataset("./data/multimodal_train_sampleset/train.sample.tsv", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)

    # ==========================
    # 3. 优化器构建
    # ==========================
    optimizer = torch.optim.AdamW(visual_projector.parameters(), lr=1e-4, weight_decay=0.05)
    total_steps = len(train_loader) * 5 # 假设训练 5 Epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    # ==========================
    # 4. 训练循环
    # ==========================
    print("Start Training Dual-Encoder (QwenText + CustomVisual)...")
    epochs = 5
    for epoch in range(epochs):
        visual_projector.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            
            roi_feat = batch['roi_features'].to(device)
            roi_box = batch['roi_boxes'].to(device)
            roi_cls = batch['roi_classes'].to(device)
            roi_mask = batch['roi_mask'].to(device)

            # --- Text Forward ---
            # input_ids 必须保持 Long 类型供 Embedding 层使用
            out = text_encoder(input_ids=input_ids, attention_mask=att_mask)
            # last_hidden_state: [batch, seq_len, hidden_dim]
            # 用 attention_mask 做 mean pooling，忽略 padding token
            hidden = out.last_hidden_state  # [batch, seq_len, hidden_dim]
            mask_exp = att_mask.unsqueeze(-1).expand(hidden.size()).float()
            text_features = (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

            # --- Image Forward ---
            image_features = visual_projector(roi_feat, roi_box, roi_cls, roi_mask) # -> [batch, 4096]

            # --- Loss Calculation ---
            loss = criterion(image_features, text_features)

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch: [{epoch+1}/{epochs}] Step: [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        print(f"==== Epoch {epoch+1} finished. Avg Loss: {total_loss/len(train_loader):.4f} ====\n")

    # 极简测例：保存训练后的图像提取层权重
    torch.save(visual_projector.state_dict(), "kdd_visual_projector_qwen3_2B.pth")
    print("Traning completed. Saved Custom Visual Projector weights.")

if __name__ == "__main__":
    train()
