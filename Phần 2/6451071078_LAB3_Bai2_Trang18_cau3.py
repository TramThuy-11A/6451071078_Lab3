import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

data = pd.read_csv('Mall_Customers.csv')

plt.figure(1, figsize=(15, 5))
sns.countplot(y='Gender', data=data)
plt.show()