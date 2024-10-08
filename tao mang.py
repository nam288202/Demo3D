import cv2
import os
import cupy as cp  # Sử dụng CuPy để chạy với GPU
import re
import matplotlib.pyplot as plt
import pandas as pd

# Đường dẫn đến thư mục chứa các ảnh
folder_path = r'D:\tepAnh'

# Lấy danh sách các file ảnh trong thư mục
image_files = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]

# Kiểm tra xem có ảnh nào trong thư mục không
if len(image_files) == 0:
    print("Không tìm thấy ảnh nào trong thư mục!")
    exit()

# Đọc ảnh đầu tiên để lấy kích thước
first_image_path = os.path.join(folder_path, image_files[0])
first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)

# Kiểm tra xem có đọc được ảnh hay không
if first_image is None:
    print("Không thể đọc ảnh từ thư mục. Hãy kiểm tra lại đường dẫn.")
    exit()

# Lấy kích thước ảnh từ ảnh đầu tiên
image_height, image_width = first_image.shape
print(f"Kích thước ảnh được xác định: {image_width} x {image_height} pixels")

# Tạo mảng 2D với kiểu dữ liệu số nguyên để lưu chỉ số ảnh có pixel sáng nhất tại mỗi vị trí
best_image_numbers = cp.full((image_height, image_width), -1, dtype=cp.int32)

# Mảng lưu giá trị pixel sáng nhất tại mỗi vị trí (tạo mảng CuPy)
max_pixel_values = cp.full((image_height, image_width), -1, dtype=cp.int32)

# Hàm để lấy phần số từ tên file và chuyển thành số nguyên
def extract_number_from_filename(filename):
    # Sử dụng biểu thức chính quy để tìm số trong tên file
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None  # Chuyển thành số nguyên

# Duyệt qua từng ảnh trong thư mục
for image_file in image_files:
    # Đường dẫn đầy đủ tới từng ảnh
    image_path = os.path.join(folder_path, image_file)

    # Đọc ảnh dưới dạng grayscale (ảnh xám)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có đúng kích thước như ảnh đầu tiên hay không
    if image.shape != (image_height, image_width):
        print(f"Kích thước ảnh {image_file} không phù hợp, bỏ qua...")
        continue

    # Chuyển đổi ảnh sang CuPy array
    image = cp.asarray(image)

    # Duyệt qua từng pixel của ảnh và cập nhật mảng giá trị pixel lớn nhất
    brighter_pixels = image > max_pixel_values
    max_pixel_values[brighter_pixels] = image[brighter_pixels]

    # Lấy số từ tên ảnh và lưu vào mảng `best_image_numbers`
    image_number = extract_number_from_filename(image_file)
    best_image_numbers[brighter_pixels] = image_number  # Gán giá trị số nguyên thay vì chuỗi

print("Mảng 2D với số ảnh có pixel sáng nhất tại mỗi vị trí đã được tạo.")

# Chuyển kết quả từ CuPy về NumPy để hiển thị
best_image_numbers_numpy = cp.asnumpy(best_image_numbers)

# Hàm lấy một mảng con tùy ý từ mảng `best_image_numbers`
def get_subarray(best_image_numbers, start_x, start_y, size):
    """
    Hàm này trích xuất một mảng con từ mảng `best_image_numbers` với kích thước `size x size`.
    start_x, start_y: Tọa độ góc trên bên trái của mảng con cần trích xuất.
    size: Kích thước của mảng con (size x size).
    """
    if (start_x + size > best_image_numbers.shape[1]) or (start_y + size > best_image_numbers.shape[0]):
        print("Lỗi: Kích thước mảng con vượt quá giới hạn của mảng gốc.")
        return None

    # Trích xuất mảng con từ `best_image_numbers`
    subarray = best_image_numbers[start_y:start_y + size, start_x:start_x + size]
    return subarray

# Ví dụ: Lấy mảng con kích thước 320x320 từ vị trí (4860, 3630)
subarray = get_subarray(best_image_numbers_numpy, 4860, 3630, 200)

# Cấu hình NumPy để in ra toàn bộ mảng mà không bị rút gọn
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print("Mảng con 320x320 từ vị trí (4860, 3630):")
print(subarray)  # In ra toàn bộ mảng con mà không bị rút gọn

# Sử dụng pandas để hiển thị mảng con dưới dạng bảng
subarray_df = pd.DataFrame(subarray)
print("Mảng con dưới dạng DataFrame:")
print(subarray_df.to_string())  # Hiển thị toàn bộ DataFrame mà không rút gọn

# Tạo hình ảnh 3D từ mảng số thứ tự ảnh
def create_3d_coordinates(subarray, scale=0.5):
    """
    Chuyển đổi mảng subarray thành tọa độ 3D.
    Mỗi giá trị trong subarray sẽ nhân với `scale` để xác định tọa độ trục z.
    """
    height, width = subarray.shape
    x_coords, y_coords, z_coords = [], [], []

    for y in range(height):
        for x in range(width):
            if subarray[y, x] != -1:  # Kiểm tra nếu giá trị khác -1
                x_coords.append(x)
                y_coords.append(y)
                # Giá trị z nhân với scale (0,5 µm)
                z_value = int(subarray[y, x]) * scale
                z_coords.append(z_value)

    return x_coords, y_coords, z_coords

# Tạo tọa độ 3D từ mảng `subarray`
x_coords, y_coords, z_coords = create_3d_coordinates(subarray, scale=0.5)

# Tạo một cửa sổ và sử dụng subplot để hiển thị cả hai hình ảnh
fig = plt.figure(figsize=(14, 6))

# Tạo vùng hiển thị 3D ở bên trái (subplot 1)
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x_coords, y_coords, z_coords, c=z_coords, cmap='viridis', marker='o')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_zlabel('Z (µm)')
ax1.set_title("3D Visualization of Pixel Values")

# Tạo vùng hiển thị 2D ở bên phải (subplot 2)
ax2 = fig.add_subplot(122)
ax2.imshow(subarray.astype(float), cmap='viridis', interpolation='nearest')
plt.colorbar(ax2.imshow(subarray.astype(float), cmap='viridis', interpolation='nearest'), ax=ax2, label='Số thứ tự ảnh')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
ax2.set_title('Hình ảnh 2D của các pixel sáng nhất tại mỗi vị trí')

# Hiển thị hình ảnh
plt.show()
