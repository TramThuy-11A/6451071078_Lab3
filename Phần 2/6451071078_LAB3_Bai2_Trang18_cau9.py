import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objs as go
import plotly.offline as py

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

data = pd.read_csv('Mall_Customers.csv')

# --- TRƯỜNG HỢP 1: Annual Income & Spending Score ---
print("=== 1. Gom cụm theo Thu nhập hằng năm và Số điểm mua sắm ===")
X2 = data[['Annual Income (k$)', 'Spending Score (1-100)']].iloc[:, :].values

inertia2 = []
for n in range(1, 11):
    algorithm2 = KMeans(n_clusters=n, init='random', random_state=42)
    algorithm2.fit(X2)
    inertia2.append(algorithm2.inertia_)

plt.figure(figsize=(10, 4))
plt.plot(np.arange(1, 11), inertia2, 'o-')
plt.title('Elbow Method cho Annual Income & Spending Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

algorithm2 = KMeans(n_clusters=5, init='random', random_state=42)
algorithm2.fit(X2)
labels2 = algorithm2.labels_
centroids2 = algorithm2.cluster_centers_

plt.figure(figsize=(10, 6))
plt.scatter(X2[:, 0], X2[:, 1], c=labels2, s=100, cmap='viridis')
plt.scatter(centroids2[:, 0], centroids2[:, 1], s=300, c='red', alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Kết quả gom cụm Annual Income & Spending Score')
plt.show()

score2 = silhouette_score(X2, labels2, metric='euclidean')
print(f"Silhouette Score (2D): {score2:.4f}\n")


# --- TRƯỜNG HỢP 2: 3D (Age, Annual Income, Spending Score) ---
print("=== 2. Gom cụm 3D: Tuổi, Thu nhập hằng năm và Số điểm mua sắm ===")
X3 = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].iloc[:, :].values

algorithm3 = KMeans(n_clusters=5, init='random', random_state=42)
algorithm3.fit(X3)
labels3 = algorithm3.labels_

data['label3'] = labels3

trace1 = go.Scatter3d(
    x=data['Age'],
    y=data['Spending Score (1-100)'],
    z=data['Annual Income (k$)'],
    mode='markers',
    marker=dict(
        color=data['label3'],
        size=10,
        line=dict(color=data['label3'], width=12),
        opacity=0.8
    )
)

dat = [trace1]
layout = go.Layout(
    title='Clusters in 3D',
    scene=dict(
        xaxis=dict(title='Age'),
        yaxis=dict(title='Spending Score'),
        zaxis=dict(title='Annual Income')
    )
)

fig = go.Figure(data=dat, layout=layout)
py.plot(fig, filename='3d_clusters.html')

score3 = silhouette_score(X3, labels3, metric='euclidean')
print(f"Silhouette Score (3D): {score3:.4f}")