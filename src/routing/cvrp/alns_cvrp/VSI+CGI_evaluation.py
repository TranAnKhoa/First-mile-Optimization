import os
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import pickle

# --- Khối code của FleetAnalyzer Class ---
# Khối này chứa logic phân tích 2 chỉ số CRR và VSI

class FleetAnalyzer:
    def __init__(self, problem_instance):
        """
        Khởi tạo FleetAnalyzer với đối tượng problem_instance (dictionary)
        đã được đọc từ file .pkl.
        """
        self.instance = problem_instance
        self.farms = problem_instance['farms']
        self.facilities = problem_instance['facilities']
        # Sử dụng 'available_trucks' hoặc 'fleet' tùy thuộc vào cấu trúc problem
        self.trucks = problem_instance.get('fleet', {}).get('available_trucks', [])
        self.dist_matrix_depot_farm = problem_instance['distance_depots_farms']
        
        # Mapping Farm ID tới Index (cần cho việc tra ma trận khoảng cách)
        self.farm_id_to_idx = problem_instance.get('farm_id_to_idx_map', {})
        
        # Mapping loại xe sang index để tra bảng binary accessibility
        # Thứ tự trong mảng accessibility giả định: [Single, 20m, 26m, Truck & Dog]
        self.type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}

        # Gom nhóm dữ liệu theo Region
        self.regions = set(f.get('region') for f in self.facilities if f.get('region'))
        self.farms_by_region = defaultdict(list)
        self.trucks_by_region = defaultdict(list)
        self.depots_by_region = defaultdict(list)

        # Phân loại Farms (gán farm vào region của depot gần nhất)
        for f in self.farms:
            f_idx = self._get_farm_idx(f['id'])
            if not self.facilities or f_idx not in self.farm_id_to_idx.values():
                 # Bỏ qua nếu không tìm thấy depot hoặc index farm
                 continue
                 
            # Tìm depot gần nhất và lấy region của nó
            closest_depot_idx = np.argmin(self.dist_matrix_depot_farm[:, f_idx])
            region = self.facilities[closest_depot_idx].get('region', 'Unknown')
            self.farms_by_region[region].append(f)

        # Phân loại Trucks và Depots
        for t in self.trucks:
            self.trucks_by_region[t.get('region', 'Unknown')].append(t)
            
        for i, d in enumerate(self.facilities):
            self.depots_by_region[d.get('region', 'Unknown')].append(i)

    def _get_farm_idx(self, fid):
        """Hàm helper để lấy index của farm, xử lý trường hợp ID có hậu tố."""
        try:
            return self.farm_id_to_idx[fid]
        except KeyError:
            # Xử lý ID ảo (ví dụ: 123_1)
            clean_id = str(fid).split('_')[0]
            # Thử với ID gốc
            if clean_id.isdigit(): 
                if int(clean_id) in self.farm_id_to_idx:
                    return self.farm_id_to_idx[int(clean_id)]
            if clean_id in self.farm_id_to_idx:
                 return self.farm_id_to_idx[clean_id]
            
            # Thử với ID dưới dạng int nếu ID gốc là string
            try:
                if int(fid) in self.farm_id_to_idx:
                    return self.farm_id_to_idx[int(fid)]
            except ValueError:
                pass

            # Fallback - trả về -1 nếu không tìm thấy, sẽ gây lỗi index nếu không xử lý
            return -1

    def _check_feasibility(self, truck, farm, depot_idx):
        """Kiểm tra tính khả thi của chuyến đi đơn (Depot -> Farm -> Depot)"""
        truck_type_idx = self.type_to_idx.get(truck['type'], -1)
        if truck_type_idx == -1: return False, 1 # Truck type không xác định

        farm_idx = self._get_farm_idx(farm['id'])
        if farm_idx == -1: return False, 4 # Farm không có index

        # 1. CHECK ACCESSIBILITY (BINARY LOOKUP)
        farm_acc = farm.get('accessibility', [1, 1, 1, 1])
        depot = self.facilities[depot_idx]
        depot_acc = depot.get('accessibility', [1, 1, 1, 1])
        
        if farm_acc[truck_type_idx] == 0 or depot_acc[truck_type_idx] == 0:
            return False, 1 

        # 2. CHECK CAPACITY
        if truck['capacity'] < farm['demand']:
            return False, 2 

        # 3. CHECK TIME FEASIBILITY
        velocity = 1.0 if truck['type'] in ["Single", "Truck and Dog"] else 0.5
        
        dist_go = self.dist_matrix_depot_farm[depot_idx, farm_idx]
        travel_time_go = dist_go / velocity
        
        farm_tw = farm.get('time_windows', {'AM': (0, 10000), 'PM': (0, 10000)}) # Fallback: rộng
        service_params = farm.get('service_time_params', (0, 1e-9))
        fix_time, var_param = service_params
        service_duration = fix_time + (farm['demand'] / var_param if var_param > 0 else 0)
        
        depot_close_time = 1900 # Giờ đóng cửa mặc định
        is_time_feasible = False
        
        for shift in ['AM', 'PM']:
            start_tw, end_tw = farm_tw.get(shift, (None, None))
            if start_tw is None or end_tw is None: continue

            arrival_at_farm = travel_time_go
            service_start = max(arrival_at_farm, start_tw)
            
            if service_start > end_tw: continue
                
            finish_service = service_start + service_duration
            dist_back = self.dist_matrix_depot_farm[depot_idx, farm_idx]
            travel_time_back = dist_back / velocity
            
            arrival_at_depot = finish_service + travel_time_back
            
            if arrival_at_depot <= depot_close_time:
                is_time_feasible = True
                break
        
        return is_time_feasible, 0 if is_time_feasible else 3

    # ================= VSI ANALYSIS (TỔNG TẢI TRỌNG) =================
    def analyze_vsi(self, safety_factor=1.1):
        results = []
        for region in self.regions:
            farms = self.farms_by_region[region]
            trucks = self.trucks_by_region[region]
            
            total_demand = sum(f['demand'] for f in farms)
            total_capacity = sum(t['capacity'] for t in trucks)
            
            vsi = total_demand / total_capacity if total_capacity > 0 else 999.0
            
            avg_cap = np.mean([t['capacity'] for t in trucks]) if trucks else 25000 
            
            shortage_mass = (total_demand * safety_factor) - total_capacity
            extra_trucks = max(0, np.ceil(shortage_mass / avg_cap))
            
            results.append({
                'Region': region,
                'Total_Demand': round(total_demand, 0),
                'Total_Capacity': round(total_capacity, 0),
                'VSI': round(vsi, 2),
                'Est_Extra_Vehicles': int(extra_trucks),
                'Status': 'CRITICAL' if vsi > 1 else ('WARNING' if vsi > 0.85 else 'OK')
            })
        
        return pd.DataFrame(results)

    # ================= CRR ANALYSIS (TÍNH KHẢ THI KỸ THUẬT) =================
    def analyze_crr(self):
        results = []
        
        for region in self.regions:
            farms = self.farms_by_region[region]
            trucks = self.trucks_by_region[region]
            depots = self.depots_by_region[region]
            
            if not farms: continue
            if not trucks:
                results.append({'Region': region, 'Total_Farms': len(farms), 'Covered_Farms': 0, 'CRR (%)': 0.0, 'Fail_Access': len(farms), 'Fail_Capacity': 0, 'Fail_Time': 0})
                continue

            covered_count = 0
            fail_reasons = defaultdict(int) # 1: Access, 2: Cap, 3: Time
            
            for farm in farms:
                is_covered = False
                reasons_for_this_farm = set()
                
                # Tìm depot gần nhất để kiểm tra tính khả thi tối ưu
                closest_depot = min(depots, key=lambda d_idx: self.dist_matrix_depot_farm[d_idx, self._get_farm_idx(farm['id'])])
                
                for truck in trucks:
                    feasible, code = self._check_feasibility(truck, farm, closest_depot)
                    
                    if feasible:
                        is_covered = True
                        break 
                    else:
                        reasons_for_this_farm.add(code)
                
                if is_covered:
                    covered_count += 1
                else:
                    # Phân loại nguyên nhân chính gây Infeasible (ưu tiên Access > Cap > Time)
                    if 1 in reasons_for_this_farm:
                         # Nếu có bất kỳ xe nào fail Access, và farm này bị bỏ, 
                         # ta cần kiểm tra xem có xe nào thoả Access không.
                        accessible_trucks = [t for t in trucks if self._check_feasibility(t, farm, closest_depot)[1] != 1]
                        if not accessible_trucks:
                            fail_reasons['Access'] += 1
                        else:
                            # Xe thoả Access nhưng vẫn fail (do Cap/Time)
                            cap_feasible = [t for t in accessible_trucks if self._check_feasibility(t, farm, closest_depot)[1] != 2]
                            if not cap_feasible:
                                fail_reasons['Capacity'] += 1
                            else:
                                fail_reasons['Time'] += 1
                    elif 2 in reasons_for_this_farm:
                        fail_reasons['Capacity'] += 1
                    elif 3 in reasons_for_this_farm:
                        fail_reasons['Time'] += 1

            crr = (covered_count / len(farms)) * 100 if farms else 0
            
            results.append({
                'Region': region,
                'Total_Farms': len(farms),
                'Covered_Farms': covered_count,
                'CRR (%)': round(crr, 1),
                'Fail_Access': fail_reasons['Access'],
                'Fail_Capacity': fail_reasons['Capacity'],
                'Fail_Time': fail_reasons['Time']
            })
            
        return pd.DataFrame(results)

    # ... (Các phần trước giữ nguyên)

    def suggest_fleet_mix(self, region, shortage_mass, trucks_in_region, farms_in_region):
        """
        Đề xuất cụ thể loại xe cần mua dựa trên đặc điểm vùng.
        Chiến thuật:
        1. Tìm xe 'Workhorse' (Xe to nhất vùng) để gánh tải chính.
        2. Tìm xe 'Access' (Xe cơ động nhất/nhỏ nhất) để cứu các farm đường khó.
        """
        if not trucks_in_region:
            # Nếu vùng chưa có xe nào, giả định lấy xe to nhất và nhỏ nhất từ toàn bộ fleet mẫu
            # Hoặc fallback về mặc định
            return {"Generic_Truck": int(shortage_mass / 25000)}

        # 1. Xác định loại xe Chủ lực (Capacity lớn nhất) và Xe Cơ động (Capacity nhỏ nhất)
        sorted_trucks = sorted(trucks_in_region, key=lambda t: t['capacity'])
        small_truck_template = sorted_trucks[0]   # Xe nhỏ nhất (ví dụ: Single)
        big_truck_template = sorted_trucks[-1]    # Xe to nhất (ví dụ: Truck and Dog)
        
        recommendation = defaultdict(int)
        remaining_shortage = shortage_mass

        # 2. Phân tích Accessibility: Có bao nhiêu demand nằm ở farm đường khó?
        # Farm đường khó = Farm mà xe to nhất KHÔNG vào được
        difficult_demand = 0
        
        # Lấy index binary của loại xe to nhất
        big_truck_type_idx = self.type_to_idx.get(big_truck_template['type'], -1)
        
        if big_truck_type_idx != -1:
            for f in farms_in_region:
                # Check access của farm với xe to nhất
                f_acc = f.get('accessibility', [1, 1, 1, 1])
                # Check access của depot gần nhất (đơn giản hoá: lấy access chung của vùng)
                # Ở đây ta check access của farm là chính
                if f_acc[big_truck_type_idx] == 0:
                    difficult_demand += f['demand']
        
        # 3. Mua xe nhỏ cho demand khó
        if difficult_demand > 0:
            count_small = int(np.ceil(difficult_demand / small_truck_template['capacity']))
            recommendation[small_truck_template['type']] += count_small
            
            # Trừ bớt lượng tải đã được gánh bởi xe nhỏ
            # (Lưu ý: xe nhỏ cũng đóng góp vào tổng capacity)
            remaining_shortage -= (count_small * small_truck_template['capacity'])

        # 4. Mua xe to cho phần thiếu hụt còn lại (Volume Shortage)
        if remaining_shortage > 0:
            count_big = int(np.ceil(remaining_shortage / big_truck_template['capacity']))
            recommendation[big_truck_template['type']] += count_big

        return dict(recommendation)

    # ================= CẬP NHẬT HÀM RUN_ANALYSIS =================
    def run_analysis(self):
        print("\n=== STARTING FLEET SUFFICIENCY ANALYSIS (SMART MIX) ===")
        
        # ... (Phần code cũ tính df_vsi và df_crr giữ nguyên) ...
        df_vsi = self.analyze_vsi()
        df_crr = self.analyze_crr()
        full_report = pd.merge(df_vsi, df_crr, on='Region')
        
        print("\n📊 DETAILED REPORT:")
        print(full_report.to_string())
        
        print("\n💡 SMART INVESTMENT STRATEGY:")
        for _, row in full_report.iterrows():
            reg = row['Region']
            vsi = row['VSI']
            
            if vsi > 1.0: # Chỉ đề xuất nếu thiếu xe
                # Tính lượng thiếu hụt (kg/lít)
                total_demand = row['Total_Demand']
                current_cap = row['Total_Capacity']
                # Safety factor 1.1
                shortage_mass = (total_demand * 1.1) - current_cap
                
                # Lấy dữ liệu trucks và farms của vùng
                trucks = self.trucks_by_region[reg]
                farms = self.farms_by_region[reg]
                
                # Gọi hàm đề xuất thông minh
                mix = self.suggest_fleet_mix(reg, shortage_mass, trucks, farms)
                
                mix_str = ", ".join([f"{qty} x {type_}" for type_, qty in mix.items()])
                print(f"- Region {reg}: Cần thêm khoảng {mix_str}")
                print(f"  (Lý do: VSI={vsi}. Ưu tiên xe nhỏ cho đường khó, xe to cho tải trọng lớn)")
            
            else:
                print(f"- Region {reg}: 🟢 Đủ xe.")

        return full_report

# --- Khối code thực thi theo yêu cầu của User ---
# --- Khối code thực thi (Thay thế phần cũ) ---
if __name__ == '__main__':
    # CẤU HÌNH ĐƯỜNG DẪN (Dùng raw string r"..." để tránh lỗi đường dẫn)
    # Hãy thay đổi đường dẫn này nếu file của bạn nằm ở chỗ khác
    INSTANCE_FILE = r"K:\Data Science\SOS lab\Project Code\output_data\CEL_instance.pkl"
    
    print(f"📂 Đang đọc instance từ: {INSTANCE_FILE}")
    
    try:
        with open(INSTANCE_FILE, 'rb') as f:
            # Load dữ liệu trực tiếp vào biến problem
            problem = pickle.load(f)
            
            # KIỂM TRA DỮ LIỆU CƠ BẢN
            if isinstance(problem, dict):
                print(f"✅ Đã đọc dữ liệu thành công! (Dạng Dictionary)")
                print(f"   - Các keys tìm thấy: {list(problem.keys())}")
                
                # Kiểm tra xem có key 'fleet' hay không (quan trọng cho FleetAnalyzer)
                if 'fleet' not in problem:
                    print("⚠️ CẢNH BÁO: Không thấy key 'fleet' trong dữ liệu. Analyzer có thể bị lỗi.")
                if 'farms' in problem:
                    print(f"   - Số lượng Farms: {len(problem['farms'])}")
                
                # --- CHẠY PHÂN TÍCH ---
                analyzer = FleetAnalyzer(problem)
                report = analyzer.run_analysis()
                
            else:
                print(f"❌ LỖI: Dữ liệu không phải là Dictionary mà là {type(problem)}. Vui lòng kiểm tra lại file tạo dữ liệu.")

    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file tại: {INSTANCE_FILE}")
    except Exception as e:
        print(f"❌ LỖI KHÔNG XÁC ĐỊNH: {e}")