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

p1 = 144.6296902,	-36.12459824


lon1,lat1 = p1
p11=lat1,lon1
p2 = 144.1315664,	-36.18881996




lon2,lat2=p2
p22=lat2,lon2
a = compute_dist (p11,p22)
print(a)
print(f"{a//60}h {(a-60*(a//60)):.0f}min")