import json
import re

# Dán đoạn log của bạn vào giữa 3 dấu nháy kép này
LOG_DATA = """
#1 [ATOMIC] Score:6160 | shaw_spatial(5%) -> regret_3
#2 [ATOMIC] Score:5256 | shaw_spatial(5%) -> regret_3
#3 [ATOMIC] Score:3555 | shaw_spatial(5%) -> regret_4
#4 [ATOMIC] Score:3035 | shaw_hybrid(5%) -> regret_4
#5 [ATOMIC] Score:2472 | shaw_temporal(5%) -> regret_4
#6 [ATOMIC] Score:136 | time_worst_removal(5%) -> regret_3
#7 [ATOMIC] Score:136 | time_worst_removal(5%) -> regret_4
#8 [SEQ] Len:3 Score:3494 | [time_worst_removal(10%)->regret_3] => [shaw_temporal(10%)->best_insertion] => [time_worst_removal(10%)->regret_4]
#9 [SEQ] Len:3 Score:2898 | [time_worst_removal(20%)->regret_2] => [shaw_spatial(5%)->regret_4] => [time_worst_removal(5%)->regret_4]
#10 [SEQ] Len:3 Score:2427 | [time_worst_removal(20%)->regret_2] => [shaw_structural(5%)->regret_2] => [time_worst_removal(10%)->regret_4]
#11 [SEQ] Len:3 Score:1999 | [time_worst_removal(20%)->regret_2] => [shaw_structural(5%)->regret_3] => [time_worst_removal(10%)->regret_2]
#12 [SEQ] Len:3 Score:1754 | [time_worst_removal(20%)->regret_2] => [shaw_spatial(5%)->regret_2] => [time_worst_removal(10%)->regret_4]
#13 [SEQ] Len:2 Score:2274 | [shaw_hybrid(5%)->regret_4] => [time_worst_removal(30%)->regret_4]
#14 [SEQ] Len:2 Score:2090 | [shaw_structural(5%)->regret_4] => [time_worst_removal(30%)->regret_4]
#15 [SEQ] Len:3 Score:1660 | [time_worst_removal(20%)->regret_4] => [shaw_temporal(5%)->regret_4] => [time_worst_removal(20%)->regret_2]
#16 [SEQ] Len:3 Score:1448 | [time_worst_removal(20%)->regret_2] => [trip_removal(5%)->regret_4] => [time_worst_removal(15%)->regret_4]
#17 [SEQ] Len:3 Score:1448 | [time_worst_removal(20%)->regret_2] => [trip_removal(5%)->best_insertion] => [time_worst_removal(30%)->regret_4]
#18 [SEQ] Len:3 Score:1448 | [time_worst_removal(5%)->regret_2] => [trip_removal(5%)->regret_2] => [time_worst_removal(10%)->regret_3]
#19 [SEQ] Len:3 Score:1448 | [time_worst_removal(20%)->regret_2] => [trip_removal(5%)->regret_4] => [time_worst_removal(10%)->regret_4]
#20 [SEQ] Len:3 Score:1448 | [time_worst_removal(20%)->regret_4] => [trip_removal(5%)->regret_4] => [time_worst_removal(10%)->regret_2]
#21 [SEQ] Len:3 Score:1448 | [time_worst_removal(20%)->regret_2] => [trip_removal(5%)->regret_4] => [time_worst_removal(10%)->regret_3]
#22 [SEQ] Len:3 Score:1448 | [time_worst_removal(20%)->regret_2] => [trip_removal(5%)->regret_4] => [time_worst_removal(10%)->regret_3]
"""

# Mapping tên sang index (Phải khớp với list trong PPO_Env)
DESTROY_OPS = ['random_removal', 'worst_removal_alpha_0', 'worst_removal_bigM', 'worst_removal_adaptive', 'time_worst_removal', 'shaw_spatial', 'shaw_hybrid', 'shaw_temporal', 'shaw_structural', 'trip_removal', 'historical_removal']
REPAIR_OPS = ['best_insertion', 'regret_2_position', 'regret_2_trip', 'regret_2_vehicle', 'regret_3_position', 'regret_3_trip', 'regret_3_vehicle', 'regret_4_position', 'regret_4_trip', 'regret_4_vehicle']
REMOVE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def get_idx(name, lst):
    for i, item in enumerate(lst):
        if item in name: return i
    # Xử lý tên tắt (ví dụ regret_2 -> regret_2_vehicle)
    # Fallback thông minh
    if 'regret_2' in name: return 3 # Default vehicle
    if 'regret_3' in name: return 6
    if 'regret_4' in name: return 9
    if 'regret_k' in name: return 3
    return 0

def get_param_idx(percent_str):
    val = int(percent_str) / 100.0
    for i, v in enumerate(REMOVE_LEVELS):
        if abs(val - v) < 0.01: return i
    return 1 # Default 10%

output_list = []

lines = LOG_DATA.strip().split('\n')
for line in lines:
    if not line.strip(): continue
    
    # Parse Rank & Info
    # #8 [SEQ] Len:3 Score:3494 | ...
    parts = line.split('|')
    info_part = parts[0]
    seq_part = parts[1].strip()
    
    rank = int(re.search(r'#(\d+)', info_part).group(1))
    
    # Tách các bước: [d->r] => [d->r]
    steps_str = seq_part.split('=>')
    
    sequence_indices = []
    sequence_pretty = []
    
    for step in steps_str:
        step = step.strip().replace('[', '').replace(']', '')
        d_part, r_part = step.split('->')
        
        d_name = d_part.split('(')[0].strip()
        param_str = re.search(r'(\d+)%', d_part).group(1)
        r_name = r_part.strip()
        
        d_idx = get_idx(d_name, DESTROY_OPS)
        p_idx = get_param_idx(param_str)
        r_idx = get_idx(r_name, REPAIR_OPS)
        
        sequence_indices.append([d_idx, p_idx, r_idx])
        sequence_pretty.append(f"[{d_name}({param_str}%)->{r_name}]")
        
    output_list.append({
        "rank": rank,
        "sequence_indices": sequence_indices,
        "sequence_pretty": sequence_pretty
    })

with open('macro_advanced_safety.json', 'w') as f:
    json.dump(output_list, f, indent=4)

print("✅ Đã tạo file JSON chuẩn. Copy vào folder Env ngay!")