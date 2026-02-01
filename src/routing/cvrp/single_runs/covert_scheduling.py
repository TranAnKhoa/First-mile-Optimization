import pandas as pd
import ast
import re
import os

# --- CẤU HÌNH ĐƯỜNG DẪN FILE ---
# Thay thế đường dẫn bên dưới bằng đường dẫn thực tế tới file CSV kết quả của bạn
INPUT_CSV_PATH = r"K:\Data Science\SOS lab\Project Code\src\routing\cvrp\CEL_decision\drl_alns_eval_0_99013_20251219_204204.csv"  
OUTPUT_EXCEL_PATH = r"K:\Data Science\SOS lab\Project Code\Check_ppo.xlsx"

def minutes_to_hhmm(minutes):
    """Chuyển đổi phút (float) sang định dạng HH:MM"""
    try:
        minutes = float(minutes)
        # Giả sử thời gian bắt đầu từ 00:00, nếu vượt quá 24h thì trừ đi
        while minutes >= 1440: minutes -= 1440
        
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"
    except:
        return "00:00"

def clean_and_parse_schedule(schedule_str):
    """Làm sạch chuỗi string chứa 'np.float64' và parse thành list"""
    if pd.isna(schedule_str): return []
    
    # Loại bỏ chữ 'np.float64(...)'
    clean_str = re.sub(r'np\.float64\((.*?)\)', r'\1', str(schedule_str))
    
    try:
        return ast.literal_eval(clean_str)
    except Exception as e:
        print(f"⚠️ Không thể đọc dòng dữ liệu này: {str(e)[:50]}...")
        return []

def process_schedule_file(input_csv, output_excel):
    if not os.path.exists(input_csv):
        print(f"❌ Lỗi: Không tìm thấy file tại {input_csv}")
        return

    print(f"📂 Đang đọc dữ liệu từ: {input_csv}")
    df = pd.read_csv(input_csv)
    
    detailed_schedule = []

    # Duyệt qua từng dòng kết quả (thường file log sẽ có nhiều dòng, ta lấy hết)
    for idx, row in df.iterrows():
        raw_schedule = row.get('solution_schedule', '[]')
        
        # Nếu file csv của bạn có nhiều dòng lịch sử, bạn có thể chỉ muốn lấy dòng tốt nhất (cuối cùng)
        # Nếu muốn lấy hết thì giữ nguyên vòng lặp này.
        
        trips = clean_and_parse_schedule(raw_schedule)
        
        # Duyệt qua từng chuyến xe trong danh sách
        for trip in trips:
            # Cấu trúc: (Depot, Truck, [Route], Shift, Start, End, Load)
            if len(trip) < 7: continue
            
            depot_idx, truck_id, route_list, shift, start_min, end_min, load = trip
            
            duration = float(end_min) - float(start_min)
            
            detailed_schedule.append({
                "Instance ID": row.get('problem_instance', idx),
                "Truck ID": truck_id,
                "Depot": depot_idx,
                "Shift": shift,
                "Start Time": minutes_to_hhmm(start_min),
                "End Time": minutes_to_hhmm(end_min),
                "Duration (min)": round(duration, 1),
                "Load (kg)": load,
                "Stops Count": len(route_list),
                "Route Sequence": " -> ".join([str(c) for c in route_list])
            })

    if not detailed_schedule:
        print("⚠️ Không tìm thấy lịch trình nào hợp lệ để xuất file.")
        return

    # Tạo DataFrame kết quả
    df_result = pd.DataFrame(detailed_schedule)
    
    # Sắp xếp: Theo Truck ID -> Thời gian xuất phát
    df_result.sort_values(by=['Truck ID', 'Start Time'], inplace=True)

    # Xuất Excel
    print(f"💾 Đang xuất {len(df_result)} chuyến đi ra file Excel: {output_excel}")
    df_result.to_excel(output_excel, index=False)
    print("✅ Hoàn tất!")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    process_schedule_file(INPUT_CSV_PATH, OUTPUT_EXCEL_PATH)