import pickle
import numpy as np # Thường dữ liệu khoa học sẽ dùng numpy

# Đường dẫn tới file .pkl bạn muốn phân tích
file_path = r"K:/Data Science/SOS lab/Project Code/code/src/routing/cvrp/data/cvrptw_testing.pkl" 

print(f"🕵️  Đang tiến hành 'mổ xẻ' file: {file_path}\n" + "="*40)

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # --- PHẦN PHÂN TÍCH DỮ LIỆU ---

    # 1. In ra kiểu dữ liệu chính của đối tượng
    print(f"\n[+] Kiểu dữ liệu chính của file: {type(data)}")

    # 2. Nếu là một dictionary (phổ biến nhất)
    if isinstance(data, dict):
        print(f"\n[+] Đây là một Dictionary với {len(data)} cặp key-value.")
        print("   Chi tiết từng key:")
        for key, value in data.items():
            print(f"\n   - Key: '{key}'")
            print(f"     - Kiểu dữ liệu của value: {type(value)}")
            
            # Nếu value là list hoặc numpy array, in thêm thông tin kích thước
            if isinstance(value, (list, np.ndarray)):
                shape = np.shape(value)
                print(f"     - Kích thước (số lượng phần tử): {shape}")
                if len(value) > 0:
                    print(f"     - Ví dụ phần tử đầu tiên: {value[0]}")
            else:
                print(f"     - Giá trị (value): {value}")

    # 3. Nếu là một list hoặc tuple
    elif isinstance(data, (list, tuple)):
        print(f"\n[+] Đây là một List/Tuple với {len(data)} phần tử.")
        if len(data) > 0:
            print(f"   - Kiểu dữ liệu của phần tử đầu tiên: {type(data[0])}")
            print(f"   - Nội dung phần tử đầu tiên: {data[0]}")

    # Trường hợp khác
    else:
        print("\n[+] Dữ liệu chi tiết:")
        print(data)


except Exception as e:
    print(f"\n❌ Đã có lỗi xảy ra: {e}")