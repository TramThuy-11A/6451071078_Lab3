import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


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

for train_index, val_index in kfold.split(train_images):
    model = create_model()
    model.fit(
        train_images[train_index],
        train_labels[train_index],
        epochs=5,
        batch_size=32,
        verbose=0,
    )
    _, val_acc = model.evaluate(
        train_images[val_index], train_labels[val_index], verbose=0
    )
    models.append(model)
    accuracy_per_fold.append(val_acc)

best_model = models[np.argmax(accuracy_per_fold)]
y_pred = np.argmax(best_model.predict(test_images), axis=1)

print(
    classification_report(
        test_labels, y_pred, target_names=[str(i) for i in range(10)]
    )
)