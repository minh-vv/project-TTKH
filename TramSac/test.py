import numpy as np
import folium
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

# Hàm khởi tạo các centroids ban đầu
def initialize_centroids(data, k):
    random_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[random_indices]
    return centroids

# Hàm gán các điểm dữ liệu vào các cụm
def assign_clusters(data, centroids):
    distances = np.zeros((data.shape[0], len(centroids)))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(data - centroid, axis=1)
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels

# Hàm cập nhật các centroids
def update_centroids(data, cluster_labels, k):
    centroids = np.array([data[cluster_labels == i].mean(axis=0) for i in range(k)])
    return centroids

# Hàm kiểm tra hội tụ
def has_converged(old_centroids, new_centroids, tolerance=1e-4):
    return np.all(np.linalg.norm(new_centroids - old_centroids, axis=1) < tolerance)

# Hàm KMeans từ đầu
def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        old_centroids = centroids
        cluster_labels = assign_clusters(data, centroids)
        centroids = update_centroids(data, cluster_labels, k)
        if has_converged(old_centroids, centroids):
            break
    return centroids, cluster_labels

# Phân cụm bằng KMeans tự xây dựng
initial_stations, cluster_labels = kmeans(data, num_charging_stations)

# Hàm tính khoảng cách địa lý bằng geopy
def geodesic_distance(a, b):
    return geodesic(a, b).km

# Hàm tính tổng khoảng cách từ các điểm đến trạm sạc gần nhất
def total_distance(stations, data):
    total_dist = 0
    for point in data:
        total_dist += min(geodesic_distance(point, centroid) for centroid in stations)
    return total_dist

# Backtracking Line Search để tìm learning rate tối ưu
def backtracking_line_search(stations, data, gradients, alpha=1.0, beta=0.5, c=1e-4):
    initial_distance = total_distance(stations, data)
    while True:
        new_stations = stations - alpha * gradients
        new_distance = total_distance(new_stations, data)
        if new_distance < initial_distance - c * alpha * np.sum(gradients ** 2):
            break
        alpha *= beta
    return alpha

# Gradient Descent với Line Search để tối ưu vị trí các trạm sạc
def gradient_descent_with_line_search(stations, data, max_iterations=60, gradient_tolerance=1e-5):
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
        print(np.linalg.norm(gradients) - gradient_tolerance)
        # Kiểm tra tiêu chí hội tụ dựa trên gradient
        if np.linalg.norm(gradients) < gradient_tolerance:
            break
        
        alpha = backtracking_line_search(stations, data, gradients)
        stations -= alpha * gradients
        print(f"đây là lần lặp thứ {i}: {total_distance(stations, data)}")
        current_distance = total_distance(stations, data)
        total_distances.append(current_distance)
    
    return stations, total_distances

# Áp dụng Gradient Descent với Line Search
optimized_stations, total_distances = gradient_descent_with_line_search(initial_stations, data, gradient_tolerance=1e-5)

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
