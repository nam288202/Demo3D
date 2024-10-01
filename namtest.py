import cv2
import os

# Nhập tọa độ pixel từ người dùng
x = int(input("Nhập tọa độ x của pixel: "))
y = int(input("Nhập tọa độ y của pixel: "))

folder_path = './'  # Đường dẫn tương đối đến thư mục chứa ảnh

# Lấy danh sách các file ảnh trong thư mục, chỉ chọn file có đuôi .bmp
image_files = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]

# Biến lưu chỉ số của ảnh có pixel sáng nhất
max_index = -1
max_pixel_value = -1

# Duyệt qua từng ảnh trong thư mục
for idx, image_file in enumerate(image_files):
    # Đường dẫn đầy đủ tới từng ảnh
    image_path = os.path.join(folder_path, image_file)

    # Mở ảnh bitmap sử dụng OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Mở ảnh ở chế độ grayscale (ảnh xám)

    # Kiểm tra nếu ảnh không được đọc thành công
    if image is None:
        print(f"Không thể mở ảnh: {image_file}")
        continue

    # Kiểm tra xem tọa độ x, y có nằm trong phạm vi ảnh không
    if x >= image.shape[1] or y >= image.shape[0]:
        print(f"Tọa độ ({x}, {y}) nằm ngoài phạm vi của ảnh {image_file}.")
        continue

    # Lấy giá trị pixel tại vị trí (x, y) mà người dùng đã nhập
    pixel_value = image[y, x]  # Chú ý rằng OpenCV sử dụng (y, x)

    # In giá trị pixel của ảnh hiện tại tại tọa độ (x, y)
    print(f"Ảnh: {image_file}, Giá trị pixel tại ({x}, {y}): {pixel_value}")

    # So sánh để tìm giá trị pixel lớn nhất
    if pixel_value >= max_pixel_value:
        max_pixel_value = pixel_value
        max_index = idx

# Kiểm tra nếu đã tìm thấy ảnh hợp lệ
if max_index != -1:
    # In ra tên của ảnh có giá trị pixel lớn nhất tại vị trí (x, y)
    best_image_name = image_files[max_index]
    print(f"\nTên ảnh có pixel sáng nhất tại vị trí ({x}, {y}) là: {best_image_name}")
    print(f"Giá trị pixel sáng nhất: {max_pixel_value}")

    # Đường dẫn đầy đủ tới ảnh có giá trị pixel lớn nhất
    best_image_path = os.path.join(folder_path, best_image_name)

    # Mở ảnh có giá trị pixel sáng nhất sử dụng OpenCV
    best_image = cv2.imread(best_image_path)

    # Hiển thị ảnh có giá trị pixel lớn nhất
    cv2.imshow(f"Ảnh có pixel sáng nhất: {best_image_name}", best_image)

    # Chờ người dùng nhấn phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Không tìm thấy ảnh hợp lệ nào.")
