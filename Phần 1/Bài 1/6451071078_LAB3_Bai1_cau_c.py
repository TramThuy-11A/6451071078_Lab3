import numpy as np
from sklearn.datasets import load_iris

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

iris = load_iris()
X = iris.data
Y = iris.target

X_scaled = X / X.max(axis=0)

print("Giá trị lớn nhất của từng cột trước chuẩn hóa:", X.max(axis=0))
print("5 mẫu dữ liệu sau khi chuẩn hóa về [0,1]:")
print(X_scaled[:5])