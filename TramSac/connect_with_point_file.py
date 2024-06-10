import folium
import numpy as np
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import matplotlib.pyplot as plt
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
initial_centroids = kmeans.cluster_centers_

# Hàm tính khoảng cách địa lý bằng geopy
def geodesic_distance(a, b):
    return geodesic(a, b).km

# Hàm tính tổng khoảng cách từ các điểm đến trạm sạc gần nhất
def total_distance(centroids, data):
    total_dist = 0
    for point in data:
        total_dist += min(geodesic_distance(point, centroid) for centroid in centroids)
    return total_dist

# Backtracking Line Search để tìm learning rate tối ưu
def backtracking_line_search(centroids, data, gradients, alpha=1.0, beta=0.5, c=1e-4):
    initial_distance = total_distance(centroids, data)
    while True:
        new_centroids = centroids - alpha * gradients
        new_distance = total_distance(new_centroids, data)
        if new_distance < initial_distance - c * alpha * np.sum(gradients ** 2):
            break
        alpha *= beta
    return alpha

# Gradient Descent với Line Search để tối ưu vị trí các trạm sạc
def gradient_descent_with_line_search(centroids, data, max_iterations=4):
    total_distances = []
    i = 0
    for _ in range(max_iterations):
        i += 1
        gradients = np.zeros_like(centroids)
        for point in data:
            distances = np.array([geodesic_distance(point, centroid) for centroid in centroids])
            closest_centroid_index = np.argmin(distances)
            closest_centroid = centroids[closest_centroid_index]
            gradients[closest_centroid_index] += (closest_centroid - point) / (distances[closest_centroid_index] + 1e-8)
        
        alpha = backtracking_line_search(centroids, data, gradients)
        centroids -= alpha * gradients
        print(f"đây là lần lặp thứ {i}: {total_distance(centroids, data)}")
        total_distances.append(total_distance(centroids, data))
    return centroids, total_distances

# Áp dụng Gradient Descent với Line Search
optimized_centroids, total_distances = gradient_descent_with_line_search(initial_centroids, data)

# Vẽ kết quả lên bản đồ bằng Folium
m = folium.Map(location=[21.028511, 105.854167], zoom_start=12)
for point in data_points:
    folium.Marker(location=point).add_to(m)

# Thay đổi này trong phần tạo marker cho trạm sạc
for centroid in optimized_centroids:
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
