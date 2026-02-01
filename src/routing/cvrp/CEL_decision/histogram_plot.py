# ============================================================
# FULL SCRIPT: Truck Usage Analysis & Visualization
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ================== CONFIG ==================
AGGREGATE_CSV_PATH = r"K:\Data Science\SOS lab\Project Code\src\routing\cvrp\CEL_after_processing\Truck Frequency.csv"

# ================== LOAD DATA ==================
df = pd.read_csv(AGGREGATE_CSV_PATH)

# Safety check
required_cols = {"Truck ID", "Frequency"}
if not required_cols.issubset(df.columns):
    raise ValueError("❌ aggregate.csv must contain columns: Truck ID, Frequency")

freq = df["Frequency"]

print("📊 Data loaded successfully")
print(df.head())

# ============================================================
# 1️⃣ HISTOGRAM (NO REORDER – STATISTICALLY CORRECT)
# ============================================================
plt.figure()
plt.hist(freq, bins=5)
plt.xlabel("Truck usage frequency")
plt.ylabel("Count")
plt.title("Histogram of Truck Usage Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# 2️⃣ HISTOGRAM + NORMAL CURVE (REFERENCE ONLY)
# ============================================================
mu, sigma = freq.mean(), freq.std()
x = np.linspace(freq.min(), freq.max(), 200)

plt.figure()
plt.hist(freq, bins=5, density=True, alpha=0.6)
plt.plot(x, norm.pdf(x, mu, sigma))
plt.xlabel("Truck usage frequency")
plt.ylabel("Density")
plt.title("Histogram with Normal Curve Overlay (Reference Only)")
plt.tight_layout()
plt.show()

# ============================================================
# 3️⃣ BAR CHART – SORTED (DECISION VIEW)
# ============================================================
df_sorted = df.sort_values("Frequency", ascending=False)

plt.figure()
plt.bar(df_sorted["Truck ID"].astype(str), df_sorted["Frequency"])
plt.xticks(rotation=90)
plt.xlabel("Truck ID (ranked)")
plt.ylabel("Total trips")
plt.title("Truck Usage Frequency (Ranked)")
plt.tight_layout()
plt.show()

# ============================================================
# 4️⃣ PARETO CHART (80–20 RULE)
# ============================================================
df_sorted["Cumulative"] = df_sorted["Frequency"].cumsum()
df_sorted["Cumulative %"] = df_sorted["Cumulative"] / df_sorted["Frequency"].sum()

fig, ax1 = plt.subplots()

ax1.bar(df_sorted["Truck ID"].astype(str), df_sorted["Frequency"])
ax1.set_xlabel("Truck ID (ranked)")
ax1.set_ylabel("Trips")

ax2 = ax1.twinx()
ax2.plot(df_sorted["Truck ID"].astype(str), df_sorted["Cumulative %"])
ax2.set_ylabel("Cumulative percentage")

plt.title("Pareto Chart – Truck Usage")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ============================================================
# DONE
# ============================================================
print("✅ All plots generated successfully.")
