import pandas as pd

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

data = pd.read_csv('Mall_Customers.csv')

print("5 dòng dữ liệu đầu tiên:")
print(data.head())

print("\nThông tin bộ dữ liệu:")
print(data.info())

print("\nSố lượng giá trị khuyết thiếu trong từng cột:")
print(data.isnull().sum())