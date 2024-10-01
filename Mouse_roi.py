import cv2
import numpy as np

isDragging = False  # Trạng thái kéo thả chuột
x0, y0, w, h = -1, -1, -1, -1  # Lưu tọa độ của vùng được chọn
blue, red = (255, 0, 0), (0, 0, 255)  # Màu sắc của hình chữ nhật (xanh dương và đỏ)
roi_list = []  # Danh sách lưu các vùng đã chọn (ROI)

scale_factor = 0.1  # Tỷ lệ resize ảnh


# Hàm resize ảnh cho hiển thị
def resize_for_display(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    if width > max_width or height > max_height:
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return image


# Hàm xử lý sự kiện chuột
def onMouse(event, x, y, flags, param):
    global isDragging, x0, y0, w, h, img_resized, img, scale_factor  # Sử dụng biến toàn cục
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

                img_draw = img_resized.copy()  # Sao chép ảnh để vẽ hình chữ nhật cuối cùng
                cv2.rectangle(img_draw, (x0, y0), (x, y), red,
                              2)  # Vẽ hình chữ nhật đỏ quanh vùng đã chọn (theo ảnh resize)

                # Lưu lại các tọa độ vùng chọn gốc để sử dụng sau này
                roi_list.append(((x0_orig, y0_orig),
                                 (x0_orig + w_orig, y0_orig + h_orig)))  # Lưu tọa độ vùng đã chọn (theo ảnh gốc)

                # Hiển thị và lưu lại vùng đã chọn theo ảnh gốc
                roi_original = img[y0_orig:y0_orig + h_orig, x0_orig:x0_orig + w_orig]  # Cắt vùng chọn theo ảnh gốc
                cv2.imshow('cropped', roi_original)  # Hiển thị vùng đã cắt (theo ảnh gốc)

                # Hiển thị ảnh với hình chữ nhật màu đỏ (theo ảnh resize)
                cv2.imshow('img', img_draw)

            else:
                cv2.imshow('img', img_resized)  # Nếu kéo thả sai hướng, hiển thị lại ảnh gốc đã resize
                print("Vui lòng kéo từ góc trên trái sang góc dưới phải.")


# Hàm hiển thị tất cả các vùng đã chọn
def show_all_rois():
    # Tạo một ảnh trống có cùng kích thước với ảnh gốc
    img_with_rois = np.zeros_like(img)
    img_with_rois[:] = (255, 255, 255)  # Đặt nền trắng cho dễ thấy

    for (start, end) in roi_list:
        cv2.rectangle(img_with_rois, start, end, red, 2)  # Vẽ hình chữ nhật đỏ cho mỗi vùng đã chọn

    img_resized_for_display = resize_for_display(img_with_rois)  # Resize cho hiển thị
    cv2.imshow('img', img_resized_for_display)  # Hiển thị tất cả các vùng đã chọn


# Đọc ảnh gốc và thay đổi kích thước ảnh
img = cv2.imread(r'D:\tepAnh\WSI_seq[101].bmp')  # Đọc ảnh gốc
img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)  # Resize ảnh

# Hiển thị ảnh đã resize
cv2.imshow('img', img_resized)

# Đăng ký sự kiện chuột
cv2.setMouseCallback('img', onMouse)

# Vòng lặp xử lý sự kiện phím nhấn
while True:
    key = cv2.waitKey(0)
    if key == ord('1'):  # Nhấn phím '1' để chọn thêm vùng mới
        print("Tiếp tục chọn vùng mới...")
    elif key == ord('0'):  # Nhấn phím '0' để dừng và hiển thị tất cả các vùng đã chọn
        print("Hiển thị tất cả các vùng đã chọn...")
        show_all_rois()
    elif key == ord('q'):  # Nhấn 'q' để thoát chương trình
        print("Đã thoát chương trình")
        break

# Đóng tất cả cửa sổ khi hoàn tất
cv2.destroyAllWindows()
