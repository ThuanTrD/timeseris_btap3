import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 1. Tạo dữ liệu chuỗi thời gian giả lập
np.random.seed(42)
t = np.arange(0, 100, 0.1)
signal = np.sin(t) + np.random.normal(0, 0.3, len(t))

plt.figure(figsize=(12, 5))
plt.plot(t, signal, label="Tín hiệu có nhiễu")
plt.plot(t, np.sin(t), label="Hàm sin chuẩn", linestyle='--')
plt.xlabel("Thời gian")
plt.ylabel("Giá trị tín hiệu")
plt.title("Biểu đồ tín hiệu sin có nhiễu")
plt.legend()
plt.show()

# 2. Tạo rolling windows (dùng để tạo đặc trưng cho GMM)
window_size = 10
X = np.array([signal[i:i+window_size] for i in range(len(signal) - window_size)])

# 3. Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Áp dụng GMM
gmm = GaussianMixture(n_components=3, random_state=0)
labels = gmm.fit_predict(X_scaled)

# 5. Trực quan hóa
plt.figure(figsize=(14, 5))
plt.plot(t, signal, label='Original Signal', alpha=0.5)
plt.scatter(t[window_size:], signal[window_size:], c=labels, cmap='viridis', s=10, label='GMM clusters')
plt.title("Phân cụm tín hiệu thời gian với GMM")
plt.legend()
plt.show()
