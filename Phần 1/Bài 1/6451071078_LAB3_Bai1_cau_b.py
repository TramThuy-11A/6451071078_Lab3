import pandas as pd
from sklearn.datasets import load_iris

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

iris = load_iris()
data = iris.data
target = iris.target

print("1. Tên các thuộc tính (feature_names):", iris.feature_names)
print("   Tên các lớp (target_names):", iris.target_names)

print(f"\n2. Kích thước dữ liệu (data.shape): {data.shape}")
print(f"   Kích thước nhãn (target.shape): {target.shape}")

print("\n3. 5 mẫu dữ liệu đầu tiên (data[:5]):")
print(data[:5])
print("   5 nhãn đầu tiên (target[:5]):", target[:5])

df = pd.DataFrame(data, columns=iris.feature_names)
print("\n4. Bảng thống kê mô tả (df.describe()):")
print(df.describe())