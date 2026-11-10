import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

iris = load_iris()
X_max = iris.data.max(axis=0)
X = iris.data / X_max
Y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
nb = GaussianNB().fit(X_train, y_train)
knn = KNeighborsClassifier().fit(X_train, y_train)

nn = Sequential(
    [
        Dense(10, activation="relu", input_shape=(4,)),
        Dense(20, activation="relu"),
        Dense(3, activation="softmax"),
    ]
)
nn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
nn.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

new_samples_raw = np.array(
    [[6.2, 2.9, 4.3, 1.3], [5.1, 3.5, 1.4, 0.2], [7.3, 2.8, 6.4, 2.1]]
)
new_samples = new_samples_raw / X_max

models = {
    "Decision Tree": dt,
    "Naive Bayes": nb,
    "KNN": knn,
    "Neural Network": nn,
}

for name, model in models.items():
    preds = model.predict(new_samples)
    if preds.ndim > 1:
        preds = np.argmax(preds, axis=1)

    labels = [iris.target_names[p] for p in preds]
    print(f"Kết quả dự đoán của mô hình [{name}]:")
    for i, label in enumerate(labels):
        print(f"  + Mẫu {i+1} {new_samples_raw[i]} => {label}")
    print()