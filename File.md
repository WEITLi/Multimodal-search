这是一份为您整理的关于数据集中涉及的所有文件、对应描述以及数据结构的详细总结：

### 1. 核心数据结构：TSV 格式（训练/验证/测试集通用）

数据集中的主要样本文件（如 `train.tsv`, `valid.tsv`, `testA.tsv`, `testB.tsv`）均采用相同的 9 列结构，以制表符（Tab）分隔。对于训练集，这些是目标（ground-truth）匹配对；对于验证和测试集，它们是供模型排序的候选对（candidate pool）。

| 列名 (Column) | 数据类型/格式 | 描述 (Description) |
| --- | --- | --- |
| **product_id** | String/Int | 商品的唯一索引ID。 |
| **image_h** | Int | 商品图片的高度。 |
| **image_w** | Int | 商品图片的宽度。 |
| **num_boxes** | Int | 图片中检测到的物体边界框（bounding boxes）数量。 |
| **boxes** | Base64 String | `[num_boxes, 4]` 的二维数组，表示每个边界框的位置（上/左/下/右）。<br>

<br>*解析方式: base64解码后转为 `float32` 的 numpy 数组。* |
| **features** | Base64 String | `[num_boxes, 2048]` 的二维数组，表示检测器提取的每个边界框的 2048 维特征。<br>

<br>*解析方式: base64解码后转为 `float32` 的 numpy 数组。* |
| **class_labels** | Base64 String | `[num_boxes]` 的一维数组，表示每个物体的类别ID（共33类）。<br>

<br>*解析方式: base64解码后转为 `int64` 的 numpy 数组。* |
| **query** | String | 与商品匹配的自然语言查询文本。 |
| **query_id** | String/Int | 查询文本的唯一索引ID。 |

---

### 2. 文件及其描述汇总表

#### 📂 训练集相关 (Training Set)

| 文件名 | 所属压缩包 | 描述 | 数据结构 |
| --- | --- | --- | --- |
| `train.tsv` | `multimodal_train.zip` | 完整的训练集，包含约 300 万对 Query 和对应的商品图片真实特征（正样本）。 | 上述 9 列 TSV 格式。 | （暂不使用）
| `train.sample.tsv` | `multimodal_train_sampleset.zip` | 训练集的采样版本，包含 1 万个样本对，适合用于快速跑通代码和调试。 | 上述 9 列 TSV 格式。 |使用
| `multimodal_labels.txt` | (未提及) | 类别映射表。包含 33 个物体分类的 Category-ID 与具体类别名称的对应关系。 | 纯文本映射关系。 |

#### 📂 验证与测试集相关 (Valid & Testing Set)

*(注：验证集约 500 个 query，测试集 A 和 B 各约 1000 个 query。每个 query 有约 30 个候选商品供排序。)*

| 文件名 | 所属压缩包 | 描述 | 数据结构 |
| --- | --- | --- | --- |
| `valid.tsv` | `multimodal_valid.zip` | 验证集候选数据。 | 上述 9 列 TSV 格式。 |
| `testA.tsv` | `multimodal_testA.zip` | 测试集 A 候选数据。 | 上述 9 列 TSV 格式。 |
| `testB.tsv` | `multimodal_testB.zip` | 测试集 B 候选数据。 | 上述 9 列 TSV 格式。 |
| `valid_answer.json` | `multimodal_valid.zip` | 验证集的标准答案（Ground-truth），列出每个 query 真正对应的商品ID（无顺序要求）。 | JSON格式：键为 `"query-id"`，值为 `["product-id", ...]` 的列表。 |
| `multimodal_validpics.zip` | (同名) | 包含验证集中涉及的 9000 张商品原始图片。 | 图片文件（如 JPG/PNG）。 |

#### 📂 评估与提交相关 (Evaluation & Submission)

| 文件名 | 所属压缩包 | 描述 | 数据结构 |
| --- | --- | --- | --- |
| `testA_answer.json` | `ans.zip` | 测试集 A 的标准答案（Ground-truth）。 | JSON格式（同 `valid_answer.json`）。 |
| `testB_answer.json` | `ans.zip` | 测试集 B 的标准答案（Ground-truth）。 | JSON格式（同 `valid_answer.json`）。 |
| `submission.csv` | (需选手生成并打包) | 模型在验证集/测试集上的预测结果提交文件。模型需要为每个 query 预测最匹配的前 5 个商品（Top-5）。 | CSV格式：包含表头 `query-id,product1,product2,product3,product4,product5`。随后每行以逗号分隔对应预测ID。 |
| (Demo文件) | `multimodal_submit_example_testA.zip` | 官方提供的测试集 A 提交流程示例文件，供选手参考提交格式。 | CSV格式（同 `submission.csv`）。 |

