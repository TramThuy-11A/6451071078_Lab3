import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X = X_train.reshape(X_train.shape[0], -1) / 255.0

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

ari = adjusted_rand_score(y_train, cluster_labels)
nmi = normalized_mutual_info_score(y_train, cluster_labels)

print("--- KẾT QUẢ ĐÁNH GIÁ GOM CỤM K-MEANS TRÊN MNIST ---")
print(f"Adjusted Rand Index (ARI):          {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print("--------------------------------------------------")

cm = confusion_matrix(y_train, cluster_labels)
df_cm = pd.DataFrame(cm, index=[f"Digit {i}" for i in range(10)], columns=[f"Cluster {i}" for i in range(10)])

print("\nBảng so sánh chi tiết giữa Nhãn thực tế và Cụm gom được (Confusion Matrix):")
print(df_cm.to_string())