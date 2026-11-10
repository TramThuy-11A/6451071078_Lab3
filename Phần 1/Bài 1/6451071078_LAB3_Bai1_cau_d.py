from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

iris = load_iris()
X = iris.data / iris.data.max(axis=0)
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f"Kích thước tập X_train: {X_train.shape}")
print(f"Kích thước tập X_test:  {X_test.shape}")
print(f"Kích thước tập y_train: {y_train.shape}")
print(f"Kích thước tập y_test:  {y_test.shape}")