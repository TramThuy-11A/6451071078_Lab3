from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0

train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

print("Kích thước train_images mới:", train_images.shape)
print("Kích thước train_labels mới:", train_labels.shape)
print("Kích thước validation_images:", val_images.shape)
print("Kích thước validation_labels:", val_labels.shape)