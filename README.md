本项目用于完成NLP课程作业。学号：25023051。

### 一、模型与依赖

- 向量提取模型: https://modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B
- 意图泛化模型: 推荐 Qwen/Qwen2.5-0.5B-Instruct 或更高版本
- 向量数据库: Milvus (Lite)

### 二、数据集 (Real-Scenario Multimodal Dataset from Taobao)
- KDD Cup 2020: https://tianchi.aliyun.com/dataset/87809

### 三、项目文件结构与功能梳理

本项目历经两版架构演进。从早期的“全表扫描纯向量比对 (V1)”，升级为了当前工业界较为主流的 **“意图微调泛化 + 多模态属性提取 + Filter-ANN双路召回 (V2)”**。

#### 通用基础功能
| 文件名 | 功能描述 |
| :--- | :--- |
| `server.py` | 启动 FastAPI 服务，加载 Qwen3-VL-Embedding-2B 模型并在本地 `8848` 暴露特征抽取接口。 |
| `check_images.py` | 验证下载的图像文件是否损坏。 |
| `File.md` | 开发说明文档，详细记录了 KDD TSV 数据集的 9 列字段含义与解析逻辑。 |

#### V1 基础版：纯密集向量召回 (Dense Retrieval)
| 文件名 | 功能描述 |
| :--- | :--- |
| `embed_to_milvus.py` | (建库) 读取本地图片，调用 `server.py` 提取向量存入 Milvus `qwen3_vl_images`。 |
| `search.py` | (检索) 命令行交互，搜文本，利用余弦相似度进行全局图片的最近邻近似搜索。 |

#### V2 进阶版：Filter-ANN 多模态 RAG 闭环 (推荐)
| 文件名 | 功能描述 |
| :--- | :--- |
| `train_kdd_multimodal.py` | [离线训练] 核心双塔模型。解析 TSV 的 2048D Box特征，通过自建 `VisualProjector` 映射为 4096D 与文字向量做对比学习(InfoNCE)。 |
| `eval_kdd.py` | [测评验证] 在 `valid.tsv` 上验证上述训练结果，计算 Recall@1, Recall@5, NDCG@5 等官方排序指标。 |
| `predict_kdd.py` | [测评提交] 对 `testA.tsv` 等测试集进行候选打分，生成官方要求的 `submission.csv`。 |
| `build_finetune_data.py` | [SFT数据] 提取 Query 词，强制 LLM Teacher 构建包含 `rewritten`(扩写) 与 `attributes`(属性) 的结构化微调数据集。 |
| `train_qwen_lora.py` | [SFT训练] 使用 LoRA 微调 Qwen-0.6B 等小模型，使其掌握意图泛化与 JSON 属性提取能力。 |
| `extract_visual_properties.py`| [属性抽取] 对图片 RoI 区域应用 CLIP，实现无文本情况下的图生类别、颜色等标签，持久化至 `item_attributes.json`。 |
| `embed_to_milvus_v2.py` | [高级入库] 创建带有 Scalar(标量字段如类别/颜色) 的新 Milvus Schema，并将向量与属性一同入库。 |
| `rag_pipeline_v2.py` | **[端到端闭环]** 运行时：1. SFT 小模型意图泛化 2. Milvus 解析出的属性下推为 Bool Filter (例如 `color=='red'`) 3. 使用 `VisualProjector` 对比检索库中同条件下特征最相近的商品。 |


### 四、V2 推荐运行执行顺序 (Execution Order)

如果你希望跑通完整的进阶版 RAG 与多模态链路，请按如下顺序执行：

**环境与数据准备段**
1. 下载 KDD 数据集解压（保证 `multimodal_train_sampleset/train.sample.tsv` 存在）。
2. `python server.py` (保持后端 Embedding 接口开启)。

**第一阶段：模型训练与对齐 (Model Alignment)**
3. `python train_kdd_multimodal.py` -> 得到投影层权重 `kdd_visual_projector_qwen3_2B.pth`。
4. `python build_finetune_data.py` -> 得到微调数据集 `qwen_structured_query_sft.jsonl`。
5. `python train_qwen_lora.py` -> 得到专注电商 Query 展开的 Qwen-LoRA 权重。

**第二阶段：特征工程与底库建设 (Feature & Indexing)**
6. `python extract_visual_properties.py` -> 生成离线商品属性库 `item_attributes.json`。
7. `python embed_to_milvus_v2.py` -> 构建带有联合倒排索引（特征 + Filter属性）的 V2 级别 Milvus 数据库。

**第三阶段：在线检索生成 (Online Interaction)**
8. `python rag_pipeline_v2.py` -> 开启命令行交互体验全新双路 Filter-ANN 检索！

*(若是仅希望参与 KDD 打榜：跳过 4~8 步，在第 3 步后依次运行 `eval_kdd.py` 观察指标，并用 `predict_kdd.py` 导出结果)。*



