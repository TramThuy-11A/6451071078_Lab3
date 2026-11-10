import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images / 255.0


def create_model():
    model = Sequential(
        [
            Flatten(input_shape=(28, 28)),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


kfold = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_per_fold = []
loss_per_fold = []

for train_index, val_index in kfold.split(train_images):
    model = create_model()
    model.fit(
        train_images[train_index],
        train_labels[train_index],
        epochs=5,
        batch_size=32,
        verbose=0,
    )
    val_loss, val_acc = model.evaluate(
        train_images[val_index], train_labels[val_index], verbose=0
    )
    accuracy_per_fold.append(val_acc)
    loss_per_fold.append(val_loss)

folds = range(1, 11)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(folds, accuracy_per_fold, "o-", color="blue")
plt.title("Accuracy sau mỗi Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.xticks(folds)

plt.subplot(1, 2, 2)
plt.plot(folds, loss_per_fold, "o-", color="red")
plt.title("Loss sau mỗi Fold")
plt.xlabel("Fold")
plt.ylabel("Loss")
plt.xticks(folds)

plt.show()