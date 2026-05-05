# %% FILTER_HIGH_VALUE_SAMPLES
import json
import pandas as pd
from pathlib import Path

# ================= 配置区 =================
INPUT_JSON = "step4_fewshot_ready.json"  # 你现有的 25k 文件
DPLD_CSV = "DPLD.csv"
OUTPUT_JSON = "fewshot_high_value.json"
TARGET_COUNT = 5000  # 目标样本量

# ================= 加载 DPLD 高频词 =================
print("📖 加载 DPLD 词典...")
df_dpld = pd.read_csv(DPLD_CSV, encoding="utf-8-sig")
# 优先保留 Context_Dependent=1 的词（高歧义），其次保留高频词
high_value_exprs = set(df_dpld[df_dpld['Context_Dependent']==1]['Expression'].astype(str).str.strip())
all_exprs = set(df_dpld['Expression'].astype(str).str.strip())

print(f"✅ 高歧义词条：{len(high_value_exprs)} 个 (如微笑、……)")
print(f"✅ 总词条数：{len(all_exprs)} 个")

# ================= 筛选逻辑 =================
print(f"📖 加载待筛选数据 ({INPUT_JSON})...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    all_data = json.load(f)

print(f"🔍 开始筛选高价值样本...")
high_value_data = []
medium_value_data = []

for item in all_data:
    text = str(item.get("query_text", ""))
    
    # 1. 优先保留：含高歧义 DPLD 词 (Context_Dependent=1)
    if any(expr in text for expr in high_value_exprs):
        high_value_data.append(item)
    # 2. 其次保留：含普通 DPLD 词
    elif any(expr in text for expr in all_exprs):
        medium_value_data.append(item)

print(f"📊 筛选结果：高歧义={len(high_value_data)}, 普通 DPLD={len(medium_value_data)}, 总={len(all_data)}")

# ================= 抽样组合 =================
final_data = high_value_data.copy()
remaining_slots = TARGET_COUNT - len(final_data)

if remaining_slots > 0 and len(medium_value_data) > 0:
    import random
    random.seed(42)
    # 从普通 DPLD 样本中随机补足
    sampled_medium = random.sample(medium_value_data, min(remaining_slots, len(medium_value_data)))
    final_data.extend(sampled_medium)
    print(f"✅ 已补足至 {len(final_data)} 条")
else:
    print(f"⚠️ 高价值样本不足 {TARGET_COUNT} 条，当前共 {len(final_data)} 条，直接使用全部")

# ================= 保存 =================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=1)

print(f"💾 已保存至 {OUTPUT_JSON}")
print(f"🎉 工作量减少：{len(all_data)} → {len(final_data)} (减少约 {100 - len(final_data)/len(all_data)*100:.1f}%)")