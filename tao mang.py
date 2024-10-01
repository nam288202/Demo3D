import cv2
import os
import numpy as np
import re  # Thêm thư viện re để xử lý chuỗi
import pandas as pd  # Thêm pandas để hiển thị mảng con dưới dạng bảng
import matplotlib.pyplot as plt  # Thêm matplotlib để vẽ hình 3D

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

# Tạo mảng 2D để lưu số của ảnh có pixel sáng nhất tại mỗi vị trí
best_image_numbers = np.empty((image_height, image_width), dtype=object)

# Mảng lưu giá trị pixel sáng nhất tại mỗi vị trí
max_pixel_values = np.full((image_height, image_width), -1)


# Hàm để lấy phần số từ tên file
def extract_number_from_filename(filename):
    # Sử dụng biểu thức chính quy để tìm số trong tên file
    match = re.search(r'\d+', filename)
    return match.group() if match else None


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

    # Duyệt qua từng pixel của ảnh và cập nhật mảng giá trị pixel lớn nhất
    brighter_pixels = image > max_pixel_values
    max_pixel_values[brighter_pixels] = image[brighter_pixels]

    # Lấy số từ tên ảnh và lưu vào mảng `best_image_numbers`
    image_number = extract_number_from_filename(image_file)
    best_image_numbers[brighter_pixels] = image_number

print("Mảng 2D với số ảnh có pixel sáng nhất tại mỗi vị trí đã được tạo.")

# Ví dụ in ra số của ảnh có pixel sáng nhất tại vị trí (100, 200)
print(f"Số của ảnh có pixel sáng nhất tại vị trí (100, 200): {best_image_numbers[100, 200]}")

# Lưu mảng kết quả thành file nếu cần thiết (ví dụ: lưu dưới dạng .npy)
output_path = os.path.join(folder_path, 'best_image_numbers.npy')
np.save(output_path, best_image_numbers)
print(f"Mảng số của ảnh đã được lưu thành công vào: {output_path}")


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


# Ví dụ: Lấy mảng con kích thước 320x320 từ vị trí (4850, 3590)
subarray = get_subarray(best_image_numbers, 4860, 3630, 200)

# 1. Cấu hình NumPy để in ra toàn bộ mảng mà không bị rút gọn
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print("Mảng con 320x320 từ vị trí (4850, 3590):")
print(subarray)  # In ra toàn bộ mảng con mà không bị rút gọn

# 2. Lưu mảng con vào file .txt để xem toàn bộ
output_txt_path = r'D:\tepAnh\subarray_320x320.txt'
np.savetxt(output_txt_path, subarray, fmt='%s', delimiter=",")
print(f"Mảng con đã được lưu vào file: {output_txt_path}")

# 3. Sử dụng pandas để hiển thị mảng con dưới dạng bảng
subarray_df = pd.DataFrame(subarray)
print("Mảng con dưới dạng DataFrame:")
print(subarray_df.to_string())  # Hiển thị toàn bộ DataFrame mà không rút gọn


# Tạo hình ảnh 3D từ mảng số tên ảnh
def create_3d_coordinates(subarray, scale=0.5):
    """
    Chuyển đổi mảng subarray thành tọa độ 3D.
    Mỗi giá trị trong subarray sẽ nhân với `scale` để xác định tọa độ trục z.
    """
    height, width = subarray.shape
    x_coords, y_coords, z_coords = [], [], []

    for y in range(height):
        for x in range(width):
            if subarray[y, x] is not None:  # Kiểm tra nếu giá trị không phải None
                x_coords.append(x)
                y_coords.append(y)
                # Giá trị z nhân với scale (0,5 µm)
                z_value = int(subarray[y, x]) * scale
                z_coords.append(z_value)

    return x_coords, y_coords, z_coords


# Tạo tọa độ 3D từ mảng `subarray`
x_coords, y_coords, z_coords = create_3d_coordinates(subarray, scale=0.5)

# Vẽ hình ảnh 3D sử dụng matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Vẽ các điểm 3D
ax.scatter(x_coords, y_coords, z_coords, c=z_coords, cmap='viridis', marker='o')

# Gán nhãn cho các trục
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_zlabel('Z (µm)')

# Hiển thị đồ thị 3D
plt.title("3D Visualization of Pixel Values")
plt.show()
