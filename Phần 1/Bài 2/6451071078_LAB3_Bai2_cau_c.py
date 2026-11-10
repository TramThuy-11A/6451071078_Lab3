from tensorflow.keras.datasets import mnist

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

print("Giá trị nhỏ nhất sau chuẩn hóa:", train_images.min())
print("Giá trị lớn nhất sau chuẩn hóa:", train_images.max())