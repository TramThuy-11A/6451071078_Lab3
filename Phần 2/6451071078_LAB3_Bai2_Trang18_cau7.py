import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

data = pd.read_csv('Mall_Customers.csv')
X1 = data[['Age', 'Spending Score (1-100)']].iloc[:, :].values

algorithm = KMeans(n_clusters=4, init='random', random_state=42)
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

plt.figure(1, figsize=(15, 7))
plt.clf()
plt.scatter(x='Age', y='Spending Score (1-100)', data=data, c=labels1, s=200, cmap='viridis')
plt.scatter(x=centroids1[:, 0], y=centroids1[:, 1], s=300, c='red', alpha=0.5)
plt.ylabel('Spending Score (1-100)')
plt.xlabel('Age')
plt.title('Clusters of Age and Spending Score')
plt.show()