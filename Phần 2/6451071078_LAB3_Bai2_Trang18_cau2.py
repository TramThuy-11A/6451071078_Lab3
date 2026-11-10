import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

data = pd.read_csv('Mall_Customers.csv')

plt.style.use('fivethirtyeight')
plt.figure(1, figsize=(15, 6))

n = 0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(data[x], bins=20, kde=True)
    plt.title('Distplot of {}'.format(x))

plt.show()