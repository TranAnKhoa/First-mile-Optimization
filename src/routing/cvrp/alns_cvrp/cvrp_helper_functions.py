import pandas as pd
import sys
import math
import random
import numpy as np
import pickle # Thêm import pickle


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]
    #Hàm này dùng để mở 1 file text và tách nội dung các file đó thành cái "từ" theo khoảng trắng - đang ko dùng

#! Hàm này dùng để đọc và xử lý dữ liệu đầu vào của bài toán từ 1 file .pkl
def read_input_cvrp(filename, instance_nr=0):
    """
    Hàm đọc dữ liệu từ file .pkl có cấu trúc dictionary và trả về dữ liệu 
    theo định dạng giống với hàm read_input_cvrp cũ.
    """
    print(f"🔄 Đang đọc file instance đã được cấu trúc: '{filename}'...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print("✅ Đã đọc dữ liệu thành công.")
    
    # --- TRÍCH XUẤT DỮ LIỆU TỪ CẤU TRÚC DICTIONARY MỚI ---
    facilities = data['facilities']
    farms = data['farms']
    fleet = data['fleet']
    
    # --- CHUẨN BỊ DỮ LIỆU THEO ĐỊNH DẠNG CŨ ---
    
    # 1. Depot (Nhà máy được coi là depot)
    depot_coords = [f['coords'] for f in facilities]
    #depot_coords là danh sách các tọa độ của nhà máy, mỗi phần tử là 1 tuple (x,y)
    # Giả định tất cả các nhà máy hoạt động 24/7. Cần điều chỉnh nếu có thông tin cụ thể.
    depot_tw = [[0, 24*60] for _ in facilities] 

    # 2. Customers (Nông trại)
    customer_coords_list = [farm['coords'] for farm in farms]
    customers_x = [coord[0] for coord in customer_coords_list]
    customers_y = [coord[1] for coord in customer_coords_list]
        
    demands = [farm['demand'] for farm in farms]
    
    # LƯU Ý: Cấu trúc cũ chỉ hỗ trợ 1 service time và 1 time window.
    # Ta sẽ phải chọn 1 trong 2 hoặc kết hợp chúng.
    # LỰA CHỌN 1: Mặc định lấy service time của nông trại.
    cus_st = [farm['service_time_params'][0] for farm in farms] # Chỉ lấy FixLoadTime
    
    # LỰA CHỌN 2: Kết hợp 2 time window (AM và PM) thành 1 time window lớn nhất.
    # Đây là một cách đơn giản hóa để bắt đầu, bạn có thể thay đổi logic này sau.
    cus_tw = []
    for farm in farms:
        start_time = farm['time_windows']['AM'][0]
        end_time = farm['time_windows']['AM'][1]
        cus_tw.append([start_time, end_time])

    # 3. Capacity
    # LƯU Ý: Cấu trúc cũ chỉ có 1 capacity. Ta sẽ lấy capacity của xe đầu tiên làm đại diện.
    # Đây là điểm cần nâng cấp trong thuật toán của bạn sau này để xử lý đội xe không đồng nhất.
    capacity = fleet['available_trucks'][0]['capacity'] if fleet['available_trucks'] else 0

    # 4. Tính toán các ma trận khoảng cách
    # Ma trận khoảng cách giữa các nông trại (customer-customer)
    distance_matrix = data['distance_matrix_farms']
    
    # Ma trận khoảng cách từ các nhà máy đến các nông trại (depot-customer)
    customer_coords_list = [farm['coords'] for farm in farms]
    
    # Gọi hàm với đúng 2 tham số là 2 danh sách tọa độ
    distance_depots = compute_distance_depots(depot_coords, customer_coords_list)
    print("distance_depots:", distance_depots, distance_depots.shape)
    
    print("🔧 Đã xử lý và chuyển đổi dữ liệu sang định dạng tương thích.")

    # Trả về dữ liệu theo đúng thứ tự và cấu trúc của hàm cũ
    # Lưu ý rằng một số dữ liệu đã được đơn giản hóa (capacity, time window)
    farm_id_to_idx_map = {farm['id']: i for i, farm in enumerate(farms)}
    data['farm_id_to_idx_map'] = farm_id_to_idx_map

    return (
        len(demands),           # nb_customers
        capacity,               # Sức chứa của 1 loại xe đại diện
        distance_matrix,        # Ma trận khoảng cách farm-farm (481x481)
        distance_depots,        # Ma trận khoảng cách facility-farm (12x481)
        demands,                # Nhu cầu của các farm
        cus_st,                 # Thời gian phục vụ tại các farm
        cus_tw,                 # Cửa sổ thời gian (đã được kết hợp) của các farm
        depot_tw,               # Cửa sổ thời gian của các facility
        data                    # **Thêm vào**: Trả về toàn bộ dữ liệu gốc để có thể truy cập các thông tin chi tiết khác
    )


# Compute the distance matrix
#! Tạo ra 1 bảng tra cứu chứa khoảng cách giữa các cặp khách hàng với nhau (tính bằng eucledian)
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def compute_distance_matrix(customers_coords):
    """Tạo ma trận khoảng cách giữa các khách hàng (nông trại)."""
    nb_customers = len(customers_coords)
    distance_matrix = np.zeros((nb_customers, nb_customers))
    for i in range(nb_customers):
        for j in range(i, nb_customers):
            coord1 = customers_coords[i]
            coord2 = customers_coords[j]
            dist = compute_dist(coord1[0], coord2[0], coord1[1], coord2[1])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix



# Compute the distances to depot
#! Tính toán khoảng cách giữa depot tới khách hàng
def compute_distance_depots(depots_coords, customers_coords):
    """
    Tính ma trận khoảng cách từ mỗi kho (nhà máy) đến mỗi khách hàng (nông trại).
    """
    nb_customers = len(customers_coords)
    nb_depots = len(depots_coords)
    distance_depots = np.zeros((nb_depots, nb_customers))
    
    for d_idx in range(nb_depots):
        depot_coord = depots_coords[d_idx]
        for c_idx in range(nb_customers):
            customer_coord = customers_coords[c_idx]
            dist = compute_dist(depot_coord[0], customer_coord[0], depot_coord[1], customer_coord[1])
            distance_depots[d_idx, c_idx] = dist
            
    return distance_depots

#! Tính eucledian giữa 2 điểm có tọa độ (xi,xj) và (yi,yj)
def compute_dist(xi, xj, yi, yj):
    """Tính khoảng cách Euclidean giữa hai điểm."""
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return exact_dist

#! Đọc số lượng xe có sẵn từ file và làm ràng buộc
def get_nb_trucks(filename):
    begin = filename.rfind("-k")
    if begin != -1:
        begin += 2
        end = filename.find(".", begin)
        return int(filename[begin:end])
    print("Error: nb_trucks could not be read from the file name. Enter it from the command line")
    sys.exit(1)
#Ví dụ problem-n50-k5.pkl thì sẽ lấy phần nằm giữa begin và end, chính là số xe, ví dụ ở đây là 5

#! TÍnh tổng tải trọng lượng hàng trên 1 route
def compute_route_load(route, demands_data):
    load = 0
    for i in route: #lặp qua từng mở khách hành trong route, vì route bắt đầu là 1-->100, nên demand phải -1
        load += demands_data[i - 1]
    return load

#! Lọc ra các khách hàng tìm năng mà xe có thể ghé thăm tiếp mà không bị quá tải
def get_customers_that_can_be_added_to_route(route_load, truck_capacity, unvisited_customers, demands_data):
    unvisited_edgible_customers = []
    for customer in unvisited_customers:
        if route_load + demands_data[customer - 1] <= truck_capacity:
            unvisited_edgible_customers.append(customer) #giúp thu hẹp phạm vi tìm kiếm để add thêm vào
    return unvisited_edgible_customers

#! Từ danh sách khách hàng hợp lệ, chọn người ở gần nhất so với vị trí hiện tại của xe
def get_closest_customer_to_add(route, unvisited_edgible_customers, dist_matrix_data, dist_depot_data):
    current_node = route[-1]
    distances = [dist_matrix_data[current_node - 1][unvisited_node - 1] for unvisited_node in
                 unvisited_edgible_customers]
    closest_customer = unvisited_edgible_customers[
        pd.Series(distances).idxmin()]  # NOTE: no -1 because this is an index, not an id
    return closest_customer

#! Tính tổng chi phí (quãng đường) của 1 giải pháp hoàn chỉnh, bao gồm tất cả các tuyến đường
def cost_routes(routes, dist_matrix_data, distance_depot_data):
    """
    Hàm này cần được điều chỉnh cẩn thận vì giờ có nhiều depot.
    Giả định mỗi route là một tuple (depot_index, [customer_list]).
    """
    cost = 0
    for depot_idx, route in routes:
        if not route:
            continue
        # Chi phí từ depot đến khách hàng đầu tiên
        cost += distance_depot_data[depot_idx][route[0] - 1]
        # Chi phí từ khách hàng cuối cùng về depot
        cost += distance_depot_data[depot_idx][route[-1] - 1]
        
        # Chi phí giữa các khách hàng
        for i in range(len(route) - 1):
            cost += dist_matrix_data[route[i] - 1][route[i + 1] - 1]
    return cost

#! Quyết định ngẫu nhiên số lượng khách hàng cần xóa
def determine_nr_nodes_to_remove(nb_customers, omega_bar_minus=5, omega_minus=0.1, omega_bar_plus=50, omega_plus=0.4):
    n_plus = min(omega_bar_plus, omega_plus * nb_customers)
    n_minus = min(n_plus, max(omega_bar_minus, omega_minus * nb_customers))
    r = random.randint(round(n_minus), round(n_plus))
    return r

#! DÙng để scale 1 tập dự liệu lại để nằm trong khoảng 0 và 1
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def update_neighbor_graph(current, new_routes, new_routes_quality):
    for route in new_routes:
        prev_node = 0
        for i in range(len(route)):
            curr_node = route[i]
            prev_edge_weight = current.graph.get_edge_weight(prev_node, curr_node)
            if new_routes_quality < prev_edge_weight:
                current.graph.update_edge(prev_node, curr_node, new_routes_quality)
            prev_node = curr_node
        prev_edge_weight = current.graph.get_edge_weight(prev_node, 0)
        if new_routes_quality < prev_edge_weight:
            current.graph.update_edge(prev_node, 0, new_routes_quality)
    return current.graph


class NeighborGraph:
    def __init__(self, num_nodes):
        self.graph = np.full((num_nodes + 1, num_nodes + 1), np.inf, dtype=np.float64)

    def update_edge(self, node_a, node_b, cost):
        # graph is kept single directional
        self.graph[node_a][node_b] = cost

    def get_edge_weight(self, node_a, node_b):
        return self.graph[node_a][node_b]
    