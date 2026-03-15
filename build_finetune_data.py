import os
import json
import time
from typing import List, Optional, Set
import pandas as pd
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────────
BASE_URL   = "https://coding.dashscope.aliyuncs.com/v1"
MODEL      = "qwen3.5-plus"
BATCH_SIZE = 20        # 每次 API 调用包含的 query 数量
MAX_RETRIES = 3        # 单次调用失败最多重试次数
RETRY_DELAY = 5        # 重试间隔（秒）

SYSTEM_PROMPT = (
    "你是一个电商搜索意图理解专家。\n"
    "我会给你一个 JSON 数组，其中每个元素是一条用户搜索词。\n"
    "请对数组中的**每一条**搜索词同时完成以下两项任务：\n"
    "1. 泛化改写：生成更详细、意图更明确的搜索短语。\n"
    "2. 属性抽取：提取商品属性（category、color 等），若未提及则留空字符串。\n\n"
    "严格要求：\n"
    "- 只输出一个合法的 JSON 数组，不要有任何多余文字或 markdown 代码块。\n"
    "- 数组长度必须与输入数组长度完全一致，顺序保持一致。\n"
    "- 每个元素格式：{\"rewritten\": \"...\", \"attributes\": {\"category\": \"...\", \"color\": \"...\"}}"
)

SFT_SYSTEM_PROMPT = (
    "你是一个电商搜索意图理解专家。请同时完成以下两项任务：\n"
    "1. 对用户的搜索词进行泛化与改写，生成更详细的搜索短语。\n"
    "2. 从用户的搜索词中提取具体商品属性（category，color）。未提及则留空。\n"
    "请严格只输出合法的 JSON，格式为：\n"
    "{\"rewritten\": \"...\", \"attributes\": {\"category\": \"...\", \"color\": \"...\"}}"
)

# ── API 客户端 ────────────────────────────────────────────────────────────────
def make_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


# ── 核心：批量调用 teacher 模型 ───────────────────────────────────────────────
def call_teacher_batch(client: OpenAI, queries: List[str]) -> Optional[List[dict]]:
    """
    将 queries 列表发给 teacher 模型，要求它在一次回复中返回等长的 JSON 数组。
    成功返回解析后的列表；失败返回 None。
    """
    user_content = json.dumps(queries, ensure_ascii=False)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()

            # 容错：去掉可能存在的 markdown 代码块标记
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            results = json.loads(raw)
            if isinstance(results, list) and len(results) == len(queries):
                return results
            else:
                print(f"  [警告] 返回长度不匹配（期望 {len(queries)}，实际 {len(results)}），重试…")
        except json.JSONDecodeError as e:
            print(f"  [错误] JSON 解析失败（第 {attempt} 次）: {e}")
        except Exception as e:
            print(f"  [错误] API 调用失败（第 {attempt} 次）: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    return None


# ── 主函数 ────────────────────────────────────────────────────────────────────
def build_finetune_data(tsv_path: str, output_jsonl: str, max_samples: int = 5000):
    """
    从 TSV 中读取 query，批量调用 teacher 模型标注，构建 SFT 数据集（chatml 格式）。
    支持断点续写：若 output_jsonl 已存在，则跳过已处理的 query。
    """
    print(f"Loading queries from {tsv_path}…")
    df = pd.read_csv(tsv_path, sep="\t", header=0)
    unique_queries = df["query"].dropna().unique()
    print(f"Found {len(unique_queries)} unique queries.")

    samples = [str(q) for q in unique_queries[:max_samples]]

    # ── 断点续写：读取已完成的 query ─────────────────────────────────────────
    done_queries = set()  # type: Set[str]
    if os.path.exists(output_jsonl):
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    # user content 就是原始 query
                    for msg in rec.get("messages", []):
                        if msg["role"] == "user":
                            done_queries.add(msg["content"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_queries)} queries already done, skipping them.")

    remaining = [q for q in samples if q not in done_queries]
    print(f"Queries to process: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    client = make_client()
    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        for batch_idx in range(total_batches):
            batch = remaining[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            print(f"  Batch {batch_idx + 1}/{total_batches}  ({len(batch)} queries)…", end=" ", flush=True)

            results = call_teacher_batch(client, batch)

            if results is None:
                print("FAILED — skipping this batch.")
                continue

            # 将每条 query + teacher 回答写入 JSONL
            for query, anno in zip(batch, results):
                assistant_content = json.dumps(anno, ensure_ascii=False)
                record = {
                    "type": "chatml",
                    "messages": [
                        {"role": "system",    "content": SFT_SYSTEM_PROMPT},
                        {"role": "user",      "content": query},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "source": "kdd_structured_query",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            out_f.flush()
            print("OK")

    # 统计最终写入行数
    with open(output_jsonl, encoding="utf-8") as f:
        total = sum(1 for _ in f)
    print(f"\n✅ Done. Total records in {output_jsonl}: {total}")


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TSV_FILE    = "./data/multimodal_train_sampleset/sample/multimodal_train_sampleset/train.sample.tsv"
    OUTPUT_FILE = "sft/qwen_structured_query_sft.jsonl"

    if not os.path.exists(TSV_FILE):
        print(f"⚠️  找不到数据文件 {TSV_FILE}")
    else:
        build_finetune_data(TSV_FILE, OUTPUT_FILE)
