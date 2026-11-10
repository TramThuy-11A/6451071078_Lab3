import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

data = pd.read_csv('Mall_Customers.csv')
X1 = data[['Age', 'Spending Score (1-100)']].iloc[:, :].values

algorithm = KMeans(n_clusters=4, init='random', random_state=42)
algorithm.fit(X1)
labels1 = algorithm.labels_

score = silhouette_score(X1, labels1, metric='euclidean')
print(f"Hệ số Dáng điệu (Silhouette Coefficient) của phân cụm (Age, Spending Score): {score:.4f}")