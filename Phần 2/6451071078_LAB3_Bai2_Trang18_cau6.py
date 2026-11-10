import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

data = pd.read_csv('Mall_Customers.csv')
X1 = data[['Age', 'Spending Score (1-100)']].iloc[:, :].values

inertia = []
for n in range(1, 11):
    algorithm = KMeans(n_clusters=n, init='random', random_state=42)
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1, 11), inertia, 'o')
plt.plot(np.arange(1, 11), inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve (Age & Spending Score)')
plt.show()

algorithm = KMeans(n_clusters=4, init='random', random_state=42)
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

print("Đã gom cụm thành công thành 4 cụm!")