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

models = []
accuracy_per_fold = []
loss_per_fold = []

fold = 1
for train_index, val_index in kfold.split(train_images):
    print(f"\n--- Đang huấn luyện Fold {fold} ---")
    model = create_model()

    X_train_fold = train_images[train_index]
    y_train_fold = train_labels[train_index]
    X_val_fold = train_images[val_index]
    y_val_fold = train_labels[val_index]

    model.fit(
        X_train_fold,
        y_train_fold,
        epochs=5,
        batch_size=32,
        validation_data=(X_val_fold, y_val_fold),
        verbose=1,
    )

    models.append(model)

    val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    accuracy_per_fold.append(val_acc)
    loss_per_fold.append(val_loss)

    print(f"Fold {fold} - Loss: {val_loss:.4f} - Accuracy: {val_acc:.4f}")
    fold += 1