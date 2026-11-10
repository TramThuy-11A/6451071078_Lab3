import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, classification_report

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

ari = adjusted_rand_score(y, cluster_labels)
nmi = normalized_mutual_info_score(y, cluster_labels)

print("=== KẾT QUẢ GOM CỤM K-MEANS TRÊN DỮ LIỆU UNG THƯ VÚ (FNA) ===")
print(f"Adjusted Rand Index (ARI):          {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print("--------------------------------------------------")

cm = confusion_matrix(y, cluster_labels)
df_cm = pd.DataFrame(cm, index=['Malignant (Ác tính)', 'Benign (Lành tính)'], columns=['Cluster 0', 'Cluster 1'])

print("\nBảng so sánh nhãn thực tế và Cụm gom được (Confusion Matrix):")
print(df_cm.to_string())