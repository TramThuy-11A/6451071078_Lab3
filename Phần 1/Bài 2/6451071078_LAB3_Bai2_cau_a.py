from tensorflow.keras.datasets import mnist

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("Đã tải thành công bộ dữ liệu MNIST từ Keras!")