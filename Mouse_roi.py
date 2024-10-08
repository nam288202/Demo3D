import cv2
import os
import cupy as cp  # Sử dụng CuPy thay cho NumPy
import re
import matplotlib.pyplot as plt

# --- Cài đặt ban đầu ---
isDragging = False  # Trạng thái kéo thả chuột
x0, y0, w, h = -1, -1, -1, -1  # Lưu tọa độ của vùng được chọn
blue, red = (255, 0, 0), (0, 0, 255)  # Màu sắc của hình chữ nhật (xanh dương và đỏ)
scale_factor = 0.1  # Tỷ lệ resize ảnh

folder_path = r'D:\tepAnh'  # Đường dẫn đến thư mục chứa các ảnh
image_files = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]  # Lấy danh sách các file ảnh trong thư mục

# --- Đọc ảnh đầu tiên để lấy kích thước ---
first_image_path = os.path.join(folder_path, image_files[0])  # Đọc ảnh đầu tiên
first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dưới dạng grayscale
image_height, image_width = first_image.shape  # Lấy kích thước của ảnh
print(f"Kích thước ảnh: {image_width} x {image_height} pixels")

# --- Hàm lấy số từ tên file ---
def extract_number_from_filename(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None  # Trả về số nguyên thay vì chuỗi

# --- Hàm hiển thị ảnh 3D ---
def create_3d_coordinates(subarray, scale=0.5):
    height, width = subarray.shape
    x_coords, y_coords, z_coords = [], [], []

    # Sử dụng CuPy để thực hiện các phép toán trên GPU
    for y in range(height):
        for x in range(width):
            if subarray[y, x] is not None:  # Kiểm tra nếu giá trị không phải None
                x_coords.append(x)
                y_coords.append(y)
                z_value = int(subarray[y, x]) * scale
                z_coords.append(z_value)

    return x_coords, y_coords, z_coords

# --- Hàm xử lý sự kiện chuột ---
def onMouse(event, x, y, flags, param):
    global isDragging, x0, y0, w, h, img_resized, img

    if event == cv2.EVENT_LBUTTONDOWN:  # Khi nhấn nút chuột trái (bắt đầu kéo thả)
        isDragging = True
        x0, y0 = x, y  # Lưu tọa độ bắt đầu (theo ảnh đã resize)
    elif event == cv2.EVENT_MOUSEMOVE:  # Khi chuột di chuyển
        if isDragging:  # Trong khi kéo thả
            img_draw = img_resized.copy()  # Sao chép ảnh gốc để vẽ hình chữ nhật tạm thời
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2)  # Vẽ hình chữ nhật xanh dương
            cv2.imshow('img', img_draw)  # Hiển thị ảnh với hình chữ nhật tạm thời
    elif event == cv2.EVENT_LBUTTONUP:  # Khi nhả nút chuột trái (kết thúc kéo thả)
        if isDragging:  # Kết thúc kéo thả
            isDragging = False
            w, h = x - x0, y - y0  # Tính toán chiều rộng và chiều cao vùng đã chọn (theo ảnh đã resize)

            if w > 0 and h > 0:  # Nếu chiều rộng và chiều cao dương (kéo đúng hướng)
                # Tính toán lại tọa độ gốc và kích thước vùng chọn theo ảnh gốc
                x0_orig = int(x0 / scale_factor)  # Chuyển đổi tọa độ x0 theo ảnh gốc
                y0_orig = int(y0 / scale_factor)  # Chuyển đổi tọa độ y0 theo ảnh gốc
                w_orig = int(w / scale_factor)  # Chuyển đổi chiều rộng theo ảnh gốc
                h_orig = int(h / scale_factor)  # Chuyển đổi chiều cao theo ảnh gốc

                # Hiển thị các tọa độ gốc tính theo ảnh gốc
                print(f"Tọa độ gốc trên ảnh gốc: x:{x0_orig}, y:{y0_orig}, w:{w_orig}, h:{h_orig}")

                # Tạo mảng best_image_numbers_roi trên GPU bằng CuPy
                best_image_numbers_roi = cp.empty((h_orig, w_orig), dtype=int)  # Thay đổi dtype thành int
                max_pixel_values_roi = cp.full((h_orig, w_orig), -1)

                # Duyệt qua từng ảnh trong thư mục để tạo best_image_numbers_roi chỉ cho vùng ROI
                for image_file in image_files:
                    image_path = os.path.join(folder_path, image_file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    # Kiểm tra kích thước ảnh
                    if image.shape != (image_height, image_width):
                        print(f"Kích thước ảnh {image_file} không phù hợp, bỏ qua...")
                        continue

                    # Lấy ROI từ ảnh
                    roi_image = image[y0_orig:y0_orig + h_orig, x0_orig:x0_orig + w_orig]

                    # Chuyển đổi ROI sang CuPy để thực hiện các phép toán trên GPU
                    roi_image_cp = cp.array(roi_image)

                    # Cập nhật max pixel values và best image numbers chỉ cho vùng ROI
                    brighter_pixels = roi_image_cp > max_pixel_values_roi
                    max_pixel_values_roi[brighter_pixels] = roi_image_cp[brighter_pixels]
                    image_number = extract_number_from_filename(image_file)
                    best_image_numbers_roi[brighter_pixels] = image_number  # Lưu số thứ tự ảnh

                # Chuyển mảng kết quả về CPU để hiển thị với Matplotlib
                best_image_numbers_roi_cpu = cp.asnumpy(best_image_numbers_roi)

                # Tạo tọa độ 3D từ mảng `best_image_numbers_roi`
                x_coords, y_coords, z_coords = create_3d_coordinates(best_image_numbers_roi_cpu, scale=0.5)

                # Tạo một cửa sổ và sử dụng subplot để hiển thị cả hai hình ảnh
                fig = plt.figure(figsize=(14, 6))

                # Tạo vùng hiển thị 3D ở bên trái (subplot 1)
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(x_coords, y_coords, z_coords, c=z_coords, cmap='viridis', marker='o')
                ax1.set_xlabel('X (pixels)')
                ax1.set_ylabel('Y (pixels)')
                ax1.set_zlabel('Z (µm)')
                ax1.set_title("3D Visualization of Pixel Values")

                # Thiết lập giới hạn cho các trục
                ax1.set_xlim(0, w_orig)  # Giới hạn trục X
                ax1.set_ylim(0, h_orig)  # Giới hạn trục Y
                ax1.set_zlim(0, max(z_coords))  # Giới hạn trục Z dựa trên giá trị tối đa

                # Đặt tỷ lệ cho các trục x, y, z
                ax1.set_box_aspect([w_orig, h_orig, max(z_coords)])  # Tỷ lệ các trục

                # Tạo vùng hiển thị 2D ở bên phải (subplot 2)
                ax2 = fig.add_subplot(122)
                ax2.imshow(best_image_numbers_roi_cpu.astype(float), cmap='viridis', interpolation='nearest')
                plt.colorbar(ax2.imshow(best_image_numbers_roi_cpu.astype(float), cmap='viridis', interpolation='nearest'), ax=ax2, label='Số thứ tự ảnh')
                ax2.set_xlabel('X (pixels)')
                ax2.set_ylabel('Y (pixels)')
                ax2.set_title('Hình ảnh 2D của các pixel sáng nhất trong ROI')

                # Lưu và hiển thị hình ảnh
                plt.show()

            else:
                cv2.imshow('img', img_resized)  # Nếu kéo thả sai hướng, hiển thị lại ảnh gốc đã resize
                print("Vui lòng kéo từ góc trên trái sang góc dưới phải.")

# --- Đọc ảnh gốc và thay đổi kích thước ảnh ---
img = cv2.imread(first_image_path)  # Đọc ảnh gốc
img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor,
                         interpolation=cv2.INTER_LINEAR)  # Resize ảnh

# Hiển thị ảnh đã resize
cv2.imshow('img', img_resized)

# Đăng ký sự kiện chuột
cv2.setMouseCallback('img', onMouse)

# Vòng lặp xử lý sự kiện phím nhấn
while True:
    key = cv2.waitKey(1)  # Thay đổi thành 1 để lặp liên tục
    if key == ord('q'):  # Nhấn 'q' để thoát chương trình
        print("Đã thoát chương trình")
        break

# Đóng tất cả cửa sổ
cv2.destroyAllWindows()
