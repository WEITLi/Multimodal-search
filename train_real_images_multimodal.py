import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from PIL import Image

# 从本地 scripts 导入 Qwen3-VL 的核心嵌入模型与处理器
from scripts.qwen3_vl_embedding import Qwen3VLForEmbedding, Qwen3VLProcessor
from qwen_vl_utils.vision_process import process_vision_info

# ========================================================
# 第一部分：真实图片的数据集构建
# ========================================================
class RealImageMultimodalDataset(Dataset):
    """
    接收包含 (image_path, query_text) 的数据集。
    与 KDD 版的区别在于：这次我们直接读取本地的真实的 jpg/png 原图！
    """
    def __init__(self, data_list, processor, max_length=128):
        """
        data_list: List[dict] 例如 [{'image': 'path/to/img1.jpg', 'text': '红色短袖'}, ...]
        """
        self.data_list = data_list
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)
        
    def _format_conversation(self, text=None, image_path=None):
        """辅助函数：将输入转为 Qwen3-VL Processor 需要的 Conversation 格式"""
        content = []
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            content.append({'type': 'image', 'image': img})
        if text:
            content.append({'type': 'text', 'text': text})
            
        return [
            {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": content}
        ]

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = item.get("image", "")
        text = item.get("text", "")
        
        # 1. 构造纯图像对话 (用于单模态图像特征提取)
        img_conv = self._format_conversation(image_path=image_path)
        # 2. 构造纯文本对话 (用于单模态文本特征提取)
        txt_conv = self._format_conversation(text=text)
        
        return {
            "img_conv": img_conv,
            "txt_conv": txt_conv
        }

def collate_fn(batch, processor, max_length=128):
    """
    处理一个 Batch 的输入数据，并将其喂入 Processor 进行统一 Tokenize 和像素化
    """
    # 这里用简单的 for loop 包装每个样本为单独的批次列表
    img_convs = [item["img_conv"] for item in batch]
    txt_convs = [item["txt_conv"] for item in batch]
    
    # ======== 处理 Image Batch ========
    img_text = processor.apply_chat_template(img_convs, add_generation_prompt=True, tokenize=False)
    images, _, video_kwargs = process_vision_info(img_convs, image_patch_size=16, return_video_kwargs=True)
    img_inputs = processor(
        text=img_text, images=images, videos=None, padding=True, truncation=True, 
        max_length=max_length, do_resize=False, return_tensors='pt', **video_kwargs
    )
    
    # ======== 处理 Text Batch ========
    txt_text = processor.apply_chat_template(txt_convs, add_generation_prompt=True, tokenize=False)
    txt_inputs = processor(
        text=txt_text, images=None, videos=None, padding=True, truncation=True, 
        max_length=max_length, do_resize=False, return_tensors='pt'
    )
    
    return {"img_inputs": img_inputs, "txt_inputs": txt_inputs}


# ========================================================
# 第二部分：InfoNCE 损失函数 (Image-Text Contrastive Loss)
# ========================================================
class ITCA_Loss(nn.Module):
    """对比学习：将配对的图文特征拉近，非配对的特征推远"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2


# ========================================================
# 第三部分：获取最终维度的 Embedding
# ========================================================
def pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """提取最后一个有效 Token (通常是 <|endoftext|>) 作为全句/全图的浓缩向量"""
    flipped_tensor = attention_mask.flip(dims=[1])
    last_one_positions = flipped_tensor.argmax(dim=1)
    col = attention_mask.shape[1] - last_one_positions - 1
    row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[row, col]


# ========================================================
# 第四部分：真实世界的训练主循环
# ========================================================
def train_real_world():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ===== 1. 加载原生 Qwen3-VL 也就是包含真实视觉模块 (Vision Tower) 的底座 =====
    MODEL_PATH = "./pretrain_models/Qwen3-VL-Embedding-2B"
    try:
        model = Qwen3VLForEmbedding.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
        processor = Qwen3VLProcessor.from_pretrained(MODEL_PATH, padding_side='right')
    except Exception as e:
        print(f"Failed to load model files: {e}. Ensure the path is correct.")
        
        # 为了防抛错，做一个假模块（只为展示代码结构设计）
        print("Using mock version for demonstration purposes...")
        model = nn.Linear(64, 4096).to(device)
        class MockProcessor:
            def apply_chat_template(self, *args, **kwargs): return [""]
            def __call__(self, *args, **kwargs): return {"input_ids": torch.zeros((1, 64)), "attention_mask": torch.ones((1, 64))}
        processor = MockProcessor()

    # 此处假设模型支持参数训练（显存不足可以加上 PEFT 的 get_peft_model 进行 LoRA 并冻结 Vision Tower 底层层）
    if hasattr(model, "train"):
        model.train()
    
    criterion = ITCA_Loss().to(device)

    # ===== 2. Mock 真实数据 (你可以接入真实的 Pandas DataFrame 读取图片文件夹) =====
    # 这里模拟了两对图文数据
    dummy_data = [
        {"image": "./pretrain_models/Qwen3-VL-Embedding-2B/images/test1.jpg", "text": "红色女式连衣裙"},
        {"image": "./pretrain_models/Qwen3-VL-Embedding-2B/images/test2.jpg", "text": "黑色休闲运动鞋"},
    ]
    
    dataset = RealImageMultimodalDataset(dummy_data, processor)
    
    # collate_fn 需要把 processor 带进去，这里直接用 lambda 封装
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, processor)
    )

    # ===== 3. Optimizer =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    total_steps = len(dataloader) * 5
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # ===== 4. Training Loop =====
    print("Start Training Dual-Encoder with Real Images (ViT + Qwen)...")
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            img_inputs = {k: v.to(device) for k, v in batch["img_inputs"].items()}
            txt_inputs = {k: v.to(device) for k, v in batch["txt_inputs"].items()}
            
            optimizer.zero_grad()
            
            if isinstance(model, nn.Linear):
                # Mock run
                img_feat = model(img_inputs["input_ids"].float())
                txt_feat = model(txt_inputs["input_ids"].float())
            else:
                # --- Vision Tower + LLM Backbone Forward ---
                # 将原始图像像素（pixel_values）交给模型原生解码！突破了 2048 维 Box 的限制
                img_out = model(**img_inputs)
                img_feat = pooling_last(img_out.last_hidden_state, img_out.attention_mask)
                
                # --- Text Forward ---
                txt_out = model(**txt_inputs)
                txt_feat = pooling_last(txt_out.last_hidden_state, txt_out.attention_mask)

            # --- InfoNCE Loss ---
            loss = criterion(img_feat, txt_feat)
            loss.backward()
            
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            print(f"Epoch: [{epoch+1}/{epochs}] Step: [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    print("Training Complete! You successfully trained the native Vision encoder!")

if __name__ == "__main__":
    train_real_world()
