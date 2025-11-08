# --- THƯ VIỆN ---
import pandas as pd
import pickle
import time
import numpy as np
import math
from collections import defaultdict

# --- 1. CẤU HÌNH ---
#EXCEL_FILE_PATH = r'E:\Syllabus for 2nd and 3rd year\cel\Project Code\output_data\Small_sample.xlsx'  # Đường dẫn tới file Excel đa sheet
#EXCEL_FILE_PATH = r'E:\Syllabus for 2nd and 3rd year\cel\Project Code\output_data\CEL_instance.xlsx'  # Đường dẫn tới file Excel đa sheet
EXCEL_FILE_PATH = r'K:\Data Science\SOS lab\Project Code\output_data\Small_sample.xlsx'
# Tên file output thống nhất
#PKL_OUTPUT_PATH = r'E:\Syllabus for 2nd and 3rd year\cel\Project Code\output_data\Small_structured_sample.pkl' 
PKL_OUTPUT_PATH = r'K:\Data Science\SOS lab\Project Code\output_data\Small_sample.pkl'
# Tên các sheet
SHEET_FACILITY_MASTER = 'FacilityMaster'
SHEET_FARM_MASTER = 'FarmMaster'
SHEET_FLEET_MASTER = 'FleetMaster'
SHEET_VARIABLE_COST = 'VariableCostByKm'
SHEET_TRUCK_LEASE = 'TruckLeaseCost'
SHEET_TRUCK_PURCHASING = 'TruckPurchasingCost'
SHEET_REGISTRATION = 'RegistrationCost'
SHEET_LABOR_Cost = 'LaborCost'
SHEET_TRANSPORT = 'TransportCost'

# --- HÀM HỖ TRỢ ---
def time_to_minutes(t):
    return t

import math

def compute_dist(coord1, coord2):
    """
    Tính khoảng cách (km) giữa hai tọa độ địa lý (vĩ độ, kinh độ)
    sử dụng công thức Haversine.

    Tham số:
      coord1: (lat1, lon1)
      coord2: (lat2, lon2)

    Trả về:
      distance_km: khoảng cách theo km
    """
    # Bán kính Trái Đất (km)
    R = 6371.0  

    # Chuyển độ sang radian
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    # Hiệu giữa vĩ độ và kinh độ
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Công thức Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = R * c
    return distance_km


# --- 2. XỬ LÝ ---
print(f"🔄 Bắt đầu đọc dữ liệu từ file đa sheet: '{EXCEL_FILE_PATH}'...")

try:
    # --- ĐỌC DỮ LIỆU TỪ CÁC SHEET ---
    df_facility_master = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_FACILITY_MASTER)
    df_farm_master = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_FARM_MASTER)
    df_fleet_master = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_FLEET_MASTER)
    df_variable_cost = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_VARIABLE_COST)
    df_truck_lease_cost = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_TRUCK_LEASE)
    df_truck_purchasing_cost = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_TRUCK_PURCHASING)
    df_registration_cost = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_REGISTRATION)
    df_labor_cost = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_LABOR_Cost)
    df_transport_cost = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_TRANSPORT)
    print("✅ Đã đọc thành công tất cả các sheet.")

    # --- TÁI CẤU TRÚC DỮ LIỆU ---
    
    # --- 1. XỬ LÝ NHÀ MÁY (FACILITIES) ---
    print("🔧 Đang xử lý dữ liệu Nhà máy...")
    facilities_list = []
    for _, row in df_facility_master.iterrows():
        facility_obj = {
            "id": row.get('FactoryRef'),
            "region": str(row.get('Region', '')).strip(),  # <-- SỬA LỖI: Dọn dẹp khoảng trắng
            "coords": [row.get('Longitude', 0.0), row.get('Latitude', 0.0)],
            "accessibility": [int(row.get(c, 0)) for c in ['TrailerSingle', 'Trailer19_20M', 'Trailer25_26M', 'TrailerTruckAndDog']],
            "service_time_params": [row.get('FixUnloadTime', 0), row.get('VarUnloadTime', 0)],
            "capacity": row.get('Capacity', 0)
        }
        facilities_list.append(facility_obj)

    # --- 2. XỬ LÝ NÔNG TRẠI (FARMS) ---
    print("🔧 Đang xử lý dữ liệu Nông trại...")
    farms_list = []
    frequency_map = {'Twice_a_Day': 2.0, '18_Hour': 24 / 18, 'Daily': 1.0, 'Skip_a_Day': 0.5}
    for _, row in df_farm_master.iterrows():
        start_am, end_am = time_to_minutes(row.get('ORDAMOpen')), time_to_minutes(row.get('ORDAMClose'))
        start_pm, end_pm = time_to_minutes(row.get('ORDPMOpen')), time_to_minutes(row.get('ORDPMClose'))
        if end_pm < start_pm and start_pm > 0: end_pm += 24 * 60
        farm_obj = {
            "id": row.get('FarmRef'),
            "region": str(row.get('Region', '')).strip(), # <-- SỬA LỖI: Dọn dẹp khoảng trắng
            "coords": [row.get('Longitude', 0.0), row.get('Latitude', 0.0)],
            "accessibility": [int(row.get(c, 0)) for c in ['TrailerSingle', 'Trailer19_20M', 'Trailer25_26M', 'TrailerTruckAndDog']],
            "time_windows": {"AM": [start_am, end_am], "PM": [start_pm, end_pm]},
            "frequency": frequency_map.get(str(row.get('UdfPickupFrequency', '')).strip(), 0),
            "demand": row.get('Demand', 0),
            "service_time_params": [row.get('FixLoadTime', 0), row.get('VarLoadTime', 0)]
        }
        farms_list.append(farm_obj)
    
    # --- 3. XỬ LÝ ĐỘI XE (FLEET) ---
    print("🔧 Đang xử lý dữ liệu Đội xe...")
    df_fleet_merged = pd.merge(df_fleet_master, df_truck_lease_cost, on='FleetRef', how='left', suffixes=('_fleet', '_cost'))
    available_trucks_list = []
    for _, row in df_fleet_merged.iterrows():
        truck_obj = {
            "id": row['FleetRef'],
            "region": str(row.get('Region_fleet', row.get('Region'))).strip(), # <-- SỬA LỖI: Dọn dẹp khoảng trắng
            "type": row['Type'],
            "capacity": row['Capacity'],
            "lease_cost_monthly": row.get('LeaseCostPerMonth', 0)
        }
        available_trucks_list.append(truck_obj)
    truck_purchasing_cost = dict(zip(df_truck_purchasing_cost['TruckType'], df_truck_purchasing_cost['PurchasingCost']))
    registration_cost = dict(zip(df_registration_cost['TruckType'], df_registration_cost['CostRate']))
    fleet_data = { "available_trucks": available_trucks_list, "purchasing_options": truck_purchasing_cost, "registration_cost_yearly": registration_cost }

    # --- 4. TÍNH TOÁN CÁC MA TRẬN KHOẢNG CÁCH ---
    print("🔧 Đang tính toán các ma trận khoảng cách...")
    farm_coords = [f['coords'] for f in farms_list]
    facility_coords = [f['coords'] for f in facilities_list]
    farm_id_to_idx_map = {farm['id']: i for i, farm in enumerate(farms_list)} #farm_id_to_idx_map = {'F001': 0, 'F002': 1, ...}

    distance_matrix_farms = np.zeros((len(farm_coords), len(farm_coords)))
    for i in range(len(farm_coords)):
        for j in range(i, len(farm_coords)):
            dist = compute_dist(farm_coords[i], farm_coords[j])
            distance_matrix_farms[i, j] = dist
            distance_matrix_farms[j, i] = dist
            
    distance_depots_farms = np.zeros((len(facility_coords), len(farm_coords)))
    for i in range(len(facility_coords)):
        for j in range(len(farm_coords)):
            dist = compute_dist(facility_coords[i], farm_coords[j])
            distance_depots_farms[i, j] = dist
    distance_matrix_depots = np.zeros((len(facility_coords), len(facility_coords)))
    for i in range(len(facility_coords)):
        for j in range(i, len(facility_coords)):
            dist = compute_dist(facility_coords[i], facility_coords[j])
            distance_matrix_depots[i, j] = dist
            distance_matrix_depots[j, i] = dist
    
    # --- 5. XỬ LÝ CHI PHÍ (COSTS) ---
    print("🔧 Đang xử lý dữ liệu Chi phí...")
    variable_cost = {(row["TruckType"].strip(), row["Region"].strip()): (row["Fuel"] + row["Tyre"] + row["Maintenance"]) for _, row in df_variable_cost.iterrows()}
    driver_info = {row["StaffID"]: row.to_dict() for _, row in df_labor_cost.iterrows()}
    transport_cost = {str(k).strip(): v for k, v in df_transport_cost.set_index('Region')['CostRate'].to_dict().items()}
    costs_data = { "variable_cost_per_km": variable_cost, "driver_costs": driver_info, "transport_coordination_cost": transport_cost }


    depot_time_windows = [[0, 24*60] for _ in facilities_list]
    # --- TỔNG HỢP VÀO CẤU TRÚC CUỐI CÙNG ---
    instance_data = { 
        "facilities": facilities_list,
        "farms": farms_list,
        "fleet": fleet_data,
        "costs": costs_data,
        "farm_id_to_idx_map": farm_id_to_idx_map,
        "distance_matrix_farms": distance_matrix_farms,
        "distance_depots_farms": distance_depots_farms,
        "depot_tw": depot_time_windows,
        "distance_matrix_depots": distance_matrix_depots
    }   
    print("{")
    for key, value in instance_data.items():
        print(f"  '{key}': ", end="") # In ra key, ví dụ: 'facilities':
        try:
            # Thử lấy 5 phần tử đầu tiên
            # (Hoạt động cho cả list, tuple và NumPy array)
            print(f"{value[:5]} ... (tổng cộng {len(value)} phần tử)")
        except (TypeError, AttributeError):
            # Nếu value không "cắt" được (VD: là 1 số, 1 dict)
            # thì cứ in ra giá trị gốc
            print(value)
    print("}")
    print("🔧 Đã tổng hợp dữ liệu (bao gồm cả ma trận khoảng cách) vào cấu trúc dictionary cuối cùng.")

    # --- LƯU FILE ---
    with open(PKL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(instance_data, f)
    
    print(f"\n🎉 Hoàn tất! Đã tạo file '{PKL_OUTPUT_PATH}' thành công.")

except Exception as e:
    import traceback
    print(f"❌ Đã xảy ra lỗi không xác định: {e}")
    traceback.print_exc()