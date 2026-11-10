import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

print("Họ và tên: Nguyễn Thuỳ Trâm - MSSV: 6451071078")
print("--------------------------------------------------")

image_path = 'dog.png'

# Tự tạo ảnh dog.png mẫu nếu chưa có file
if not os.path.exists(image_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor('skyblue')
    circle1 = plt.Circle((0.5, 0.5), 0.3, color='brown')
    circle2 = plt.Circle((0.35, 0.65), 0.08, color='black')
    circle3 = plt.Circle((0.65, 0.65), 0.08, color='black')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

image = Image.open(image_path).convert('RGB')

imarr = np.array(image)
rows = imarr.shape[0]
cols = imarr.shape[1]

data = imarr / 255.0
data = data.reshape(rows * cols, 3)

kmeans = KMeans(n_clusters=32, max_iter=300, random_state=42)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

img_recolored = new_colors.reshape(imarr.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(image)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(img_recolored)
ax[1].set_title('32-color Image', size=16)

plt.show()

img_recolored_save = img_recolored * 255
new_img = Image.fromarray(img_recolored_save.astype(np.uint8))
new_img.save('compressed_img2.png')

print("Đã giảm số màu và lưu file compressed_img2.png thành công!")