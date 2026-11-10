import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
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
class_prec = [[], [], []]
class_rec = [[], [], []]
class_f1 = [[], [], []]

for train_index, val_index in kfold.split(X):
    model = create_model()
    model.fit(X[train_index], Y[train_index], epochs=100, batch_size=1, verbose=0)
    
    y_pred = np.argmax(model.predict(X[val_index], verbose=0), axis=1)
    p, r, f1, _ = precision_recall_fscore_support(Y[val_index], y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    
    for c in range(3):
        class_prec[c].append(p[c])
        class_rec[c].append(r[c])
        class_f1[c].append(f1[c])

df_res = pd.DataFrame({
    'Class': iris.target_names,
    'Average Precision': [np.mean(class_prec[c]) for c in range(3)],
    'Average Recall': [np.mean(class_rec[c]) for c in range(3)],
    'Average F1-score': [np.mean(class_f1[c]) for c in range(3)]
})

print("--- ĐÁNH GIÁ TRUNG BÌNH THEO TỪNG LỚP (PER CLASS) ---")
print(df_res.to_string(index=False))