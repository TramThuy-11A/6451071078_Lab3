import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

iris = load_iris()
X = iris.data / iris.data.max(axis=0)
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
acc_list, prec_list, rec_list, f1_list = [], [], [], []

for train_index, val_index in kfold.split(X):
    model = create_model()
    model.fit(X[train_index], Y[train_index], epochs=100, batch_size=1, verbose=0)
    
    y_pred = np.argmax(model.predict(X[val_index], verbose=0), axis=1)
    acc_list.append(accuracy_score(Y[val_index], y_pred))
    p, r, f1, _ = precision_recall_fscore_support(Y[val_index], y_pred, average='macro', zero_division=0)
    prec_list.append(p)
    rec_list.append(r)
    f1_list.append(f1)

print("--- ĐÁNH GIÁ TRUNG BÌNH CỦA MÔ HÌNH (AVERAGE METRICS) ---")
print(f"Average Accuracy:  {np.mean(acc_list):.4f}")
print(f"Average Precision: {np.mean(prec_list):.4f}")
print(f"Average Recall:    {np.mean(rec_list):.4f}")
print(f"Average F1-score:  {np.mean(f1_list):.4f}")