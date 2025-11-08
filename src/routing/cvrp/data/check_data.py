import pickle
import pandas as pd
import os

# --- THAY ĐỔI CÁC THAM SỐ NÀY NẾU CẦN ---
FILE_PATH = 'code/src/routing/cvrp/data/cvrp_100_10000.pkl'
INSTANCE_INDEX = 0 # Ví dụ, bạn có thể thay đổi
# -----------------------------------------

print(f"🔍 Đang tiến hành đọc file: {FILE_PATH}")

try:
    with open(FILE_PATH, 'rb') as f:
        all_data = pickle.load(f)

    print(f"✅ Đọc file thành công! File chứa tổng cộng {len(all_data)} instances.")

    if INSTANCE_INDEX >= len(all_data) or INSTANCE_INDEX < 0:
        print(f"❌ Lỗi: Instance_index {INSTANCE_INDEX} không hợp lệ. Vui lòng chọn một số từ 0 đến {len(all_data) - 1}.")
    else:
        instance_data = all_data[INSTANCE_INDEX]
        depot_coords = instance_data[0]
        customer_coords = instance_data[1]
        demands = instance_data[2]
        capacity = instance_data[3]

        print(f"\n--- Đang xử lý instance {INSTANCE_INDEX} ---")

        # --- CHUẨN BỊ DỮ LIỆU ĐỂ LƯU ---

        # 1. Depot Coordinates
        df_depot = pd.DataFrame([depot_coords], columns=['Depot_X', 'Depot_Y'])

        # 2. Customer Coordinates & Demands
        customer_data_list = []
        for i in range(len(customer_coords)):
            customer_data_list.append({
                'Customer_ID': i,
                'Coord_X': customer_coords[i][0],
                'Coord_Y': customer_coords[i][1],
                'Demand': demands[i]
            })
        df_customers = pd.DataFrame(customer_data_list)

        # 3. Capacity
        df_capacity = pd.DataFrame([{'Vehicle_Capacity': capacity}])

        # --- LƯU VÀO MỘT FILE EXCEL VỚI NHIỀU SHEETS ---
        output_dir = 'output_data'
        os.makedirs(output_dir, exist_ok=True)
        
        output_excel_filename = os.path.join(output_dir, f'cvrp_instance_{INSTANCE_INDEX}_combined.xlsx')

        print(f"\n--- Đang lưu dữ liệu của instance {INSTANCE_INDEX} vào {output_excel_filename} ---")

        with pd.ExcelWriter(output_excel_filename, engine='xlsxwriter') as writer:
            df_depot.to_excel(writer, sheet_name='Depot_Coordinates', index=False)
            df_customers.to_excel(writer, sheet_name='Customer_Data', index=False)
            df_capacity.to_excel(writer, sheet_name='Vehicle_Capacity', index=False)
        
        print(f"✅ Đã lưu tất cả dữ liệu vào file Excel: {output_excel_filename}")
        print("   - Sheet 'Depot_Coordinates': Chứa tọa độ điểm xuất phát.")
        print("   - Sheet 'Customer_Data': Chứa ID, tọa độ và nhu cầu của khách hàng.")
        print("   - Sheet 'Vehicle_Capacity': Chứa sức chứa của xe.")


except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file tại đường dẫn '{FILE_PATH}'.")
    print("  - Vui lòng kiểm tra lại đường dẫn và đảm bảo bạn chạy script từ thư mục gốc của dự án.")
except Exception as e:
    print(f"❌ Đã xảy ra lỗi không xác định: {e}")