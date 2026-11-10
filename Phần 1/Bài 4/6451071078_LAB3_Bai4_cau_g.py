import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

iris = load_iris()
X_max = iris.data.max(axis=0)
X = iris.data / X_max
Y = iris.target

def create_model():
    model = Sequential([
        Input(shape=(4,)),
        Dense(10, activation='relu'),
        Dense(20, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
models = []
f1_scores = []

for train_index, val_index in kfold.split(X):
    model = create_model()
    model.fit(X[train_index], Y[train_index], epochs=100, batch_size=1, verbose=0)
    
    y_pred = np.argmax(model.predict(X[val_index], verbose=0), axis=1)
    _, _, f1, _ = precision_recall_fscore_support(Y[val_index], y_pred, average='macro', zero_division=0)
    
    models.append(model)
    f1_scores.append(f1)

best_idx = np.argmax(f1_scores)
best_model = models[best_idx]
print(f"Mô hình tốt nhất thuộc Fold {best_idx + 1} với Macro F1-score: {f1_scores[best_idx]:.4f}\n")

new_samples_raw = np.array([
    [6.2, 2.9, 4.3, 1.3],
    [5.1, 3.5, 1.4, 0.2],
    [7.3, 2.8, 6.4, 2.1]
])
new_samples = new_samples_raw / X_max

preds = np.argmax(best_model.predict(new_samples, verbose=0), axis=1)
labels = [iris.target_names[p] for p in preds]

print("Dự đoán nhãn cho 03 mẫu dữ liệu mới:")
for i, label in enumerate(labels):
    print(f"  + Mẫu {i+1} {new_samples_raw[i]} => {label}")