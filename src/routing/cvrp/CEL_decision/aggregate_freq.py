import pandas as pd
import os
from collections import Counter

# ================== CONFIG ==================
INPUT_FOLDER = r"K:\Data Science\SOS lab\Project Code\src\routing\cvrp\CEL_after_processing"
OUTPUT_FILE = os.path.join(INPUT_FOLDER, "aggregate.csv")

# ================== CORE LOGIC ==================
def aggregate_truck_frequency(input_folder, output_file):
    files = [f for f in os.listdir(input_folder) if f.endswith("_after.csv")]

    if not files:
        print("❌ No *_after.csv files found.")
        return

    print(f"🚀 Aggregating from {len(files)} files...\n")

    truck_counter = Counter()

    for file_name in files:
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)

        if "Truck ID" not in df.columns:
            print(f"⚠️ Skip {file_name} (no Truck ID column)")
            continue

        # mỗi dòng = 1 trip → count thẳng
        truck_counter.update(df["Truck ID"])

    # Convert to DataFrame
    df_agg = (
        pd.DataFrame(truck_counter.items(), columns=["Truck ID", "Frequency"])
        .sort_values("Frequency", ascending=False)
        .reset_index(drop=True)
    )

    df_agg.to_csv(output_file, index=False)

    print(f"✅ Aggregate file saved at:\n{output_file}")
    print(f"📊 Total trucks: {len(df_agg)}")

# ================== RUN ==================
if __name__ == "__main__":
    aggregate_truck_frequency(INPUT_FOLDER, OUTPUT_FILE)
