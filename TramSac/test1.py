import numpy as np
import folium
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import csv

# Đọc dữ liệu từ file CSV
data_points = []
with open('hanoi_hotspots.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Bỏ qua hàng tiêu đề
    for row in reader:
        data_points.append((float(row[0]), float(row[1])))

# Chuyển đổi dữ liệu thành numpy array
data = np.array(data_points)

# Số lượng trạm sạc
num_charging_stations = 10

# Phân cụm bằng KMeans
kmeans = KMeans(n_clusters=num_charging_stations, random_state=0).fit(data)
initial_stations = kmeans.cluster_centers_

# Hàm tính khoảng cách địa lý bằng geopy
def geodesic_distance(a, b):
    return geodesic(a, b).km

# Hàm tính tổng khoảng cách từ các điểm đến trạm sạc gần nhất
def total_distance(stations, data):
    total_dist = 0
    for point in data:
        total_dist += min(geodesic_distance(point, centroid) for centroid in stations)
    return total_dist

# Gradient Descent với Line Search để tối ưu vị trí các trạm sạc
def gradient_descent_with_line_search(stations, data, learning_rate=0., max_iterations=1000, gradient_tolerance=1e-5):
    total_distances = []
    i = 0
    for _ in range(max_iterations):
        i += 1
        gradients = np.zeros_like(stations)
        for point in data:
            distances = np.array([geodesic_distance(point, centroid) for centroid in stations])
            closest_centroid_index = np.argmin(distances)
            closest_centroid = stations[closest_centroid_index]
            gradients[closest_centroid_index] += (closest_centroid - point) / (distances[closest_centroid_index] + 1e-8)
        
        # Kiểm tra tiêu chí hội tụ dựa trên gradient
        if np.linalg.norm(gradients) < gradient_tolerance:
            break
        
        stations -= learning_rate * gradients
        current_distance = total_distance(stations, data)
        total_distances.append(current_distance)
        print(f"Iteration {i}: Total Distance = {current_distance}")
    
    return stations, total_distances

# Áp dụng Gradient Descent với Line Search
optimized_stations, total_distances = gradient_descent_with_line_search(initial_stations, data, learning_rate=0.0001, gradient_tolerance=1e-5)

# Vẽ kết quả lên bản đồ bằng Folium
m = folium.Map(location=[21.028511, 105.854167], zoom_start=12)
for point in data_points:
    folium.Marker(location=point).add_to(m)

# Thay đổi để sử dụng biểu tượng trạm sạc
for centroid in optimized_stations:
    folium.Marker(
        location=centroid, 
        icon=folium.Icon(icon='charging-station', prefix='fa', color='red')
    ).add_to(m)

# Lưu bản đồ thành file HTML
m.save('charging_stations_map.html')

# Vẽ đồ thị tổng khoảng cách giảm dần qua mỗi lần lặp
plt.plot(total_distances)
plt.xlabel('Iteration')
plt.ylabel('Total Distance (km)')
plt.title('Total Distance vs Iteration')
plt.show()

# Hiển thị bản đồ
m
