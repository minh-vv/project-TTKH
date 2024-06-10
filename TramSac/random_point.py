import numpy as np
import csv

# Hà Nội nằm trong khoảng tọa độ sau
latitude_min = 20.90
latitude_max = 21.10
longitude_min = 105.75
longitude_max = 105.95

# Tạo ngẫu nhiên 100 tọa độ trong khoảng này
num_points = 50
latitudes = np.random.uniform(latitude_min, latitude_max, num_points)
longitudes = np.random.uniform(longitude_min, longitude_max, num_points)
data_points = list(zip(latitudes, longitudes))

# Lưu dữ liệu các điểm vào file CSV
with open('hanoi_hotspots.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['latitude', 'longitude'])
    writer.writerows(data_points)
