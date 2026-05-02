# %% STEP4_LLM_BATCH_ANNOTATION_FIXED
import os
import json
import re
import time
import pandas as pd
from pathlib import Path
import ollama
from tqdm import tqdm

# ================= 配置区 =================
# 自动获取脚本所在目录，避免路径错误
SCRIPT_DIR = Path(__file__).parent.resolve()
FEWSHOT_JSON = SCRIPT_DIR / "fewshot_high_value.json"  # 确保文件名与 Colab 输出一致
DPLD_CSV = "C:/Alina/Research/version2/dpld_pipeline/DPLD.csv"
OUTPUT_CSV = SCRIPT_DIR / "augmented_dataset_v1.csv"
PROGRESS_FILE = SCRIPT_DIR / "progress_43.json"

MODEL = "qwen2.5:7b"          # 显存充足可换 :14b
BATCH_SIZE = 12               # Ollama 稳定并发数
CONFIDENCE_THRESHOLD = 0.75  # 仅保留高置信预测
SAVE_INTERVAL = 300          # 每处理多少条保存一次进度

# ================= 嵌入式 DPLD 匹配器 =================
class DPLDMatcher:
    def __init__(self, csv_path):
        self.rules = []
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
                # 仅加载上下文依赖型 (Context_Dependent=1) 的词条
                dep_df = df[df['Context_Dependent'] == 1][['Expression', 'Primary_Function']]
                for _, row in dep_df.iterrows():
                    self.rules.append({
                        "expr": str(row['Expression']).strip(),
                        "func": row['Primary_Function']
                    })
                print(f"✅ DPLD 动态规则已加载 | 有效词条：{len(self.rules)}")
            except Exception as e:
                print(f"⚠️ DPLD 加载失败：{e}")
        else:
            print(f"⚠️ 未找到 DPLD 文件：{csv_path}")

    def match_comment(self, text):
        hits = []
        for rule in self.rules:
            if rule['expr'] in text:
                hits.append(rule['expr'])
        return hits

# 初始化匹配器
dpld_matcher = DPLDMatcher(DPLD_CSV)

# ================= Prompt 模板 =================
PROMPT_TEMPLATE = """你是中文社交媒体语用学专家。请一次性判断下方列表中的 {count} 条评论。

【DPLD 判定优先规则】
1. 若评论包含"微笑""……""？？？""doge"等符号，且同时出现反问句、否定词、连续标点或负面共现词，则强制标记 intent=1（隐性阴阳），即使字面无攻击性。
2. 若仅含常规表情/语气词且无语境违和，标记 intent=0；Offensiveness 依实际敌意程度判定。
3. 严禁仅凭字面意思下结论，必须结合语用反转逻辑。

【输出要求】
- 仅返回合法 JSON 列表，不包含任何解释、Markdown 或前后缀
- 每个对象必须包含：{{"req_id": int, "intent": int, "offensiveness": int, "confidence": float}}
- req_id: 对应待判列表中的序号（1, 2, 3...）
- intent: 0=显性，1=隐性阴阳
- offensiveness: 0=非冒犯，1=冒犯/歧视/偏见
- confidence: 0.0~1.0（模型把握度）

【相似范例参考】
{seeds_prompt}

【待判评论列表】
{comments_text}

【你的判断】
"""

# ================= 工具函数 =================
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_ids": [], "results": []}

def save_progress(processed_ids, results):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_ids": processed_ids, "results": results}, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"💾 进度已保存：{len(results)} 条有效样本 | {PROGRESS_FILE.name}")

def safe_parse_json(text):
    """清洗 LLM 可能包裹的 Markdown 或多余字符，提高解析成功率"""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        parts = re.split(r"```", text, maxsplit=2)
        if len(parts) > 1:
            text = parts[1].strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    # 智能提取第一个 [ 到最后一个 ] 之间的内容
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# ================= 主流程 =================
def main():
    if not FEWSHOT_JSON.exists():
        raise FileNotFoundError(f"❌ 找不到检索文件：{FEWSHOT_JSON}\n请确保 step4_fewshot_ready.json 已在同一目录下")

    print(f"📖 正在加载检索结果... ({FEWSHOT_JSON})")
    with open(FEWSHOT_JSON, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    df_ret = pd.DataFrame(raw_list)
    total = len(df_ret)
    print(f"✅ 加载成功！共 {total} 条记录\n")

    progress = load_progress()
    processed_set = set(progress["processed_ids"])
    all_results = progress["results"]

    pending = df_ret[~df_ret["query_id"].isin(processed_set)]
    print(f"🚀 开始 Qwen2.5 本地批量标注 ({len(pending)} 条待处理)...\n")

    # 分批处理
    batches = [pending.iloc[i:i+BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    
    for i, batch_df in enumerate(tqdm(batches, desc="⏳ 批注进度")):
        batch_records = batch_df.to_dict(orient="records")
        
        # --- 核心修改：构建"多合一"Prompt ---
        # 1. 拼接待判评论
        comments_text = ""
        id_map = {}  # 用于映射索引和原始 ID
        for idx, item in enumerate(batch_records):
            cid = item["query_id"]
            text = str(item["query_text"])[:300]  # 截断防超长
            
            # 检测 DPLD 动态词并追加标记
            hits = dpld_matcher.match_comment(text)
            dpld_tag = f" ⚠️[动态词:{','.join(hits)}]" if hits else ""
            
            comments_text += f"{idx+1}.{dpld_tag} {text}\n"
            id_map[idx+1] = item
        
        # 2. 准备 Few-shot (取该批次第一条数据的种子作为范例)
        first_item = batch_records[0]
        valid_seeds = first_item.get("top5_seeds", [])[:2]  # 只取 2 个种子做参考
        seeds_prompt = "\n".join([f"- Hostile={k['hostile']} | {k['text'][:50]}..." for k in valid_seeds])

        # 3. 组装最终 Prompt
        full_prompt = PROMPT_TEMPLATE.format(
            count=len(batch_records),
            seeds_prompt=seeds_prompt,
            comments_text=comments_text
        )
        
        # --- 核心修改：只发 1 次请求 ---
        try:
            # num_predict 调大，因为要返回多个 JSON 对象
            resp = ollama.chat(
                model=MODEL, 
                messages=[{"role":"user", "content": full_prompt}], 
                format="json", 
                options={"temperature": 0.1, "num_predict": 3000} 
            )
            
            parsed = safe_parse_json(resp["message"]["content"])
            
            # 解析返回的列表
            if isinstance(parsed, list):
                for item_res in parsed:
                    req_id = item_res.get("req_id")  # 获取我们在 Prompt 里标的 ID
                    if req_id in id_map:
                        original_item = id_map[req_id]
                        cid = original_item["query_id"]
                        
                        conf = float(item_res.get("confidence", 0.0))
                        if conf >= CONFIDENCE_THRESHOLD:
                            all_results.append({
                                "comment_id": cid,
                                "reply_text": str(original_item["query_text"])[:400],
                                "intent": int(item_res["intent"]),
                                "offensiveness": int(item_res["offensiveness"]),
                                "confidence": round(conf, 2),
                                "method": "Batch_Inference_DPLD"
                            })
                            processed_set.add(cid)
        except Exception as e:
            print(f"批次 {i} 请求失败：{e}")
            # 失败可以选择重试或跳过
            
        # 定期保存
        if len(all_results) % (SAVE_INTERVAL * 2) == 0 and len(all_results) > 0:
             save_progress(list(processed_set), all_results)
        
        # 保护本地 GPU/CPU
        time.sleep(0.5)

    # 最终保存
    save_progress(list(processed_set), all_results)
    
    # 终端统计报告
    if len(all_results) > 0:
        intents = pd.Series([r["intent"] for r in all_results]).value_counts()
        print(f"\n{'='*50}")
        print(f"✅ 标注完成！有效样本：{len(all_results)} / {total}")
        print(f"📊 Intent 分布：显性 (0)={intents.get(0, 0)}, 隐性 (1)={intents.get(1, 0)}")
        print(f"💾 完整数据已保存至：{OUTPUT_CSV}")
        print(f"{'='*50}")
    else:
        print("\n⚠️ 未生成任何有效样本，请检查置信度阈值或 LLM 输出格式")

if __name__ == "__main__":
    main()