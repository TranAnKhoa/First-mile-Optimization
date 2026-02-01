import pandas as pd
import ast
import re
import os

# ================== CẤU HÌNH FOLDER ==================
INPUT_FOLDER = r"K:\Data Science\SOS lab\Project Code\src\routing\cvrp\CEL_decision"
OUTPUT_FOLDER = r"K:\Data Science\SOS lab\Project Code\src\routing\cvrp\CEL_after_processing"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================== HELPER FUNCTIONS ==================
def minutes_to_hhmm(minutes):
    try:
        minutes = float(minutes)
        while minutes >= 1440:
            minutes -= 1440
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"
    except:
        return "00:00"


def clean_and_parse_schedule(schedule_str):
    if pd.isna(schedule_str):
        return []

    clean_str = re.sub(r'np\.float64\((.*?)\)', r'\1', str(schedule_str))

    try:
        return ast.literal_eval(clean_str)
    except Exception as e:
        print(f"⚠️ Parse fail: {str(e)[:60]}...")
        return []

# ================== CORE PROCESS ==================
def process_single_csv(input_csv, output_csv):
    print(f"📂 Processing: {os.path.basename(input_csv)}")
    df = pd.read_csv(input_csv)

    detailed_schedule = []

    for idx, row in df.iterrows():
        trips = clean_and_parse_schedule(row.get("solution_schedule", "[]"))

        for trip in trips:
            if len(trip) < 7:
                continue

            depot_idx, truck_id, route_list, shift, start_min, end_min, load = trip
            duration = float(end_min) - float(start_min)

            detailed_schedule.append({
                "Instance ID": row.get("problem_instance", idx),
                "Truck ID": truck_id,
                "Depot": depot_idx,
                "Shift": shift,
                "Start Time": minutes_to_hhmm(start_min),
                "End Time": minutes_to_hhmm(end_min),
                "Duration (min)": round(duration, 1),
                "Load (kg)": load,
                "Stops Count": len(route_list),
                "Route Sequence": " -> ".join(map(str, route_list))
            })

    if not detailed_schedule:
        print("⚠️ No valid trips found, skipped.")
        return

    df_out = pd.DataFrame(detailed_schedule)
    df_out.sort_values(by=["Truck ID", "Start Time"], inplace=True)

    df_out.to_csv(output_csv, index=False)
    print(f"✅ Saved: {os.path.basename(output_csv)}")


def process_all_files():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".csv")]

    if not files:
        print("❌ No CSV files found.")
        return

    print(f"🚀 Found {len(files)} files. Start processing...\n")

    for file_name in files:
        input_path = os.path.join(INPUT_FOLDER, file_name)
        base_name = os.path.splitext(file_name)[0]
        output_name = f"{base_name}_after.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        process_single_csv(input_path, output_path)

    print("\n🎉 ALL FILES DONE!")

# ================== RUN ==================
if __name__ == "__main__":
    process_all_files()
