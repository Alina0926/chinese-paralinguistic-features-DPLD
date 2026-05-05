import pandas as pd
import json
import re
import time
import os
import ollama
from tqdm import tqdm
from pathlib import Path

# ================= 1. 配置区 =================
MODEL = "qwen2.5:14b"
INPUT_CSV = "dpld_seeds_cleaned_for_llm.csv"
OUTPUT_CSV = "DPLD_annotated_v1.csv"
OUTPUT_DIR = "dpld_pipeline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 优化配置
NUM_PREDICT = 2048
NUM_CTX = 4096
BATCH_SIZE = 1

PROGRESS_JSON = os.path.join(OUTPUT_DIR, "annotation_progress.json")
BACKUP_CSV = os.path.join(OUTPUT_DIR, "DPLD_annotated_progress.csv")

# ================= 2. Prompt 模板 =================
PROMPTS = {
    "Bilibili_Emotion": """你是计算语言学与中文社交媒体语用学专家。请严格基于B站语境标注。
输出必须为JSON数组格式，例如：[{{"Expression":"笑哭","Synonyms":["..."],"Pragmatic_Function":"Exaggeration","Intensity_Score":2,"Usage_Example":"..."}}]
约束：
1. 仅输出纯JSON，不要任何解释、Markdown标记
2. Pragmatic_Function 仅限：Sarcasm, Aggression, Defense, Exaggeration, Neutral
3. Intensity_Score 为1-5整数
4. 强度参考：微笑(3) | doge(4) | 大哭(2)
输入列表：{items}""",

    "Punctuation_Pattern": """你是计算语言学与中文社交媒体语用学专家。请标注连续标点模式。
输出必须为JSON数组格式，例如：[{{"Expression":"……","Synonyms":["..."],"Pragmatic_Function":"Sarcasm","Intensity_Score":3,"Usage_Example":"..."}}]
约束：
1. 仅输出纯JSON，不要任何解释
2. Pragmatic_Function 仅限：Sarcasm, Aggression, Defense, Exaggeration, Neutral
3. Intensity_Score 为1-5整数
4. 强度参考：……(3) | ？？？(4) | ！！！(2)
输入列表：{items}""",

    "Slang": """你是计算语言学与中文社交媒体语用学专家。请基于中文互联网语境标注。
若词在常规语境中逻辑不通（如"吃瓜"=旁观、"典"=刻板、"画饼"=空头承诺），则判定为网络隐喻。
输出必须为JSON数组格式，例如：[{{"Expression":"典","Synonyms":["..."],"Pragmatic_Function":"Aggression","Intensity_Score":3,"Usage_Example":"..."}}]
约束：
1. 仅输出纯JSON，不要任何解释
2. Pragmatic_Function 仅限：Sarcasm, Aggression, Defense, Exaggeration, Neutral
3. Intensity_Score 为1-5整数
4. 强度参考：典(3) | 急了(4) | 差不多得了(2)
输入列表：{items}"""
}

# ================= 3. 辅助函数 =================
def safe_parse_json(text):
    """兼容单对象和数组两种格式"""
    text = text.strip()
    
    # 清洗 Markdown
    if text.startswith("```"):
        text = re.split(r"```", text)[1].strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    
    # 提取 JSON
    match = re.search(r'(\[.*?\]|\{.*?\})', text, re.DOTALL)
    if match:
        text = match.group(1)
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        elif isinstance(parsed, list):
            return parsed
        else:
            return None
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON解析失败")
        return None

def load_progress():
    if os.path.exists(PROGRESS_JSON):
        with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_exprs": [], "results": [], "failed_batches": []}

def save_progress(progress):
    with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    pd.DataFrame(progress["results"]).to_csv(BACKUP_CSV, index=False, encoding="utf-8-sig")

# ================= 4. 主流程 =================
print("🚀 开始 Qwen2.5-14B 本地批量标注...")
print(f"📊 配置：Batch Size={BATCH_SIZE}, num_predict={NUM_PREDICT}\n")

# 🔧 关键修复：彻底清洗 CSV 列名和数据
print("📂 读取 CSV 文件...")
df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')

# 打印原始列名（显示 repr 以便看到隐藏字符）
print(f" 原始列名（repr）: {repr(df.columns.tolist())}")

# 彻底清洗列名：去除所有空白字符、引号、不可见字符
def clean_column_name(col):
    # 转换为字符串
    col = str(col)
    # 去除首尾空白
    col = col.strip()
    # 去除各种引号
    col = col.strip('"').strip("'").strip('`')
    # 去除不可见字符（如零宽空格、BOM等）
    col = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', col)
    # 再次去除空白
    col = col.strip()
    return col

df.columns = [clean_column_name(col) for col in df.columns]
print(f"✅ 清洗后列名: {df.columns.tolist()}")

# 验证必需列
required_cols = ["Expression", "Type", "Frequency"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ CSV缺少必需列: {missing_cols}。当前列名: {df.columns.tolist()}")

# 清洗数据：去除空行和空值
print("🧹 清洗数据...")
df = df.dropna(subset=["Expression", "Type"])  # 删除 Expression 或 Type 为空的行
df["Expression"] = df["Expression"].astype(str).str.strip()  # 去除首尾空格
df = df[df["Expression"] != ""]  # 删除空字符串
df = df[df["Expression"] != "nan"]  # 删除 "nan" 字符串

print(f"📊 清洗后总行数: {len(df)}")

progress = load_progress()
completed = set(progress["completed_exprs"])
results = progress["results"]

print(f"\n📁 总词条数: {len(df)}")
print(f"✅ 已完成: {len(completed)}")
print(f"⏳ 待处理: {len(df) - len(completed)}\n")

for type_name, group in df.groupby("Type"):
    prompt_tpl = PROMPTS.get(type_name, PROMPTS["Slang"])
    
    # 🔧 关键修复：正确提取 Expression 列为纯字符串列表
    exprs = group["Expression"].dropna().astype(str).str.strip().tolist()
    # 过滤掉空字符串和 "nan"
    exprs = [e for e in exprs if e and e != "nan" and e not in completed]
    pending = exprs
    
    if not pending:
        print(f"⏭️ {type_name} 已全部完成，跳过。")
        continue
    
    print(f"\n🔍 开始处理 {type_name} ({len(pending)} 条)...")
    
    # 批次处理
    for i in tqdm(range(0, len(pending), BATCH_SIZE), desc="  进度"):
        batch = pending[i:i+BATCH_SIZE]
        
        # 🔧 关键修复：batch 现在是纯字符串列表
        try:
            batch_json = json.dumps(batch, ensure_ascii=False)
            prompt = prompt_tpl.format(items=batch_json)
        except Exception as e:
            print(f"\n  ❌ Prompt格式化失败: {e}")
            print(f"  批次内容: {batch}")
            continue
        
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": 0.2,
                    "num_predict": NUM_PREDICT,
                    "num_ctx": NUM_CTX
                }
            )
            
            parsed = safe_parse_json(response["message"]["content"])
            
            if isinstance(parsed, list) and len(parsed) == len(batch):
                for item in parsed:
                    item["Type"] = type_name
                    results.append(item)
                    completed.add(item["Expression"])
                
                progress["completed_exprs"] = list(completed)
                progress["results"] = results
                save_progress(progress)
            else:
                print(f"\n  ⚠️ 数量不匹配: 期望{len(batch)}个，实际{len(parsed) if parsed else 0}个")
                
        except Exception as e:
            print(f"\n  ❌ 请求异常: {e}")
            progress["failed_batches"].append({
                "type": type_name,
                "batch": batch,
                "error": str(e)
            })
        
        time.sleep(0.5)

# ================= 5. 最终导出 =================
df_final = pd.DataFrame(results)
df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n{'='*60}")
print(f"✅ 标注完成！")
print(f"📊 总输出: {len(df_final)} 条")
print(f"📁 保存至: {OUTPUT_CSV}")
print(f"\n📊 语用功能分布:")
print(df_final["Pragmatic_Function"].value_counts())
print(f"\n📊 强度分布:")
print(df_final["Intensity_Score"].value_counts().sort_index())
print(f"{'='*60}")
