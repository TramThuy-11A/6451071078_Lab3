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

mean_acc = np.mean(accuracy_per_fold)
std_acc = np.std(accuracy_per_fold)
mean_loss = np.mean(loss_per_fold)
std_loss = np.std(loss_per_fold)

print(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
print(f"Mean Loss:     {mean_loss:.4f} (+/- {std_loss:.4f})")