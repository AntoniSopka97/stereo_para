import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import numpy as np
import sys

# 1. Загрузка оригинальной калибровки
if not os.path.exists("stereo_calibration.npz"):
    print("Ошибка: Нет файла stereo_calibration.npz!")
    sys.exit(1)

calib_data = np.load("stereo_calibration.npz")
map_l_x, map_l_y = calib_data["map_l_x"], calib_data["map_l_y"]
map_r_x, map_r_y = calib_data["map_r_x"], calib_data["map_r_y"]
Q = calib_data["Q"]  # Работаем строго с ОРИГИНАЛЬНОЙ матрицей Q 1920x1080

cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)
for cap in [cap0, cap2]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

window_name = "Stereo Measurer (Space to Freeze, Click to Measure)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

# Глобальные переменные для стоп-кадра
points_3d = None
frozen_image = None

# Функция обработки кликов мыши с оконным фильтром 7x7
def click_event(event, x, y, flags, param):
    global points_3d, frozen_image
    if event == cv2.EVENT_LBUTTONDOWN and points_3d is not None:
        
        # Пересчитываем координаты клика из окна 960x540 в физическую матрицу 1920x1080
        full_x = int(x * 2)
        full_y = int(y * 2)
        
        H, W, _ = points_3d.shape
        window_size = 3  # Радиус окна (размер 7х7 пикселей в full-разрешении)
        
        # Границы окна вокруг оригинального пикселя
        y_min, y_max = max(0, full_y - window_size), min(H, full_y + window_size)
        x_min, x_max = max(0, full_x - window_size), min(W, full_x + window_size)
        
        sub_matrix = points_3d[y_min:y_max, x_min:x_max]
        z_values = sub_matrix[:, :, 2]
        
        # Вырезаем битые точки (бесконечности, наны)
        valid_mask = (~np.isnan(z_values)) & (~np.isinf(z_values)) & (z_values > 100) & (z_values < 5000)
        valid_z = z_values[valid_mask]
        
        if len(valid_z) == 0:
            print(f"[-] Точка [{x}, {y}]: Ошибка сопоставления. Нет текстуры. Кликните в контрастное место.")
            return
            
        # Считаем устойчивую медиану по окну
        Z_mm = np.median(valid_z)
        X_mm = np.median(sub_matrix[:, :, 0][valid_mask])
        Y_mm = np.median(sub_matrix[:, :, 1][valid_mask])
        
        distance_cm = Z_mm / 10.0
        print(f"[+] Клик в [{x}, {y}] -> Миллиметры в 3D: X={X_mm:.1f}, Y={Y_mm:.1f} | РАССТОЯНИЕ: {distance_cm:.2f} см")
        
        # Обновляем экран
        img_copy = frozen_image.copy()
        cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img_copy, f"{distance_cm:.1f} cm", (x + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, img_copy)

cv2.setMouseCallback(window_name, click_event)

# Конфигурация матчеров под FULLHD
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0, 
    numDisparities=256, # На 1920х1080 диапазон поиска сдвига должен быть шире!
    blockSize=5,
    P1=8*3*5**2, 
    P2=32*3*5**2, 
    disp12MaxDiff=2, 
    uniquenessRatio=15,
    speckleWindowSize=100, 
    speckleRange=2, 
    mode=cv2.StereoSGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(10000.0)
wls_filter.setSigmaColor(1.5)

is_frozen = False
print("\n--- ИНСТРУКЦИЯ ИЗМЕРЕНИЯ (ВЕРСИЯ 2) ---")
print("1. Направьте стереопару на объект.")
print("2. Нажмите ПРОБЕЛ для фиксации четкого СТОП-КАДРА.")
print("3. Кликайте мышкой — глубина считается по оригинальной калибровочной матрице.")

while True:
    if not is_frozen:
        ret0, frame0 = cap0.read()
        ret2, frame2 = cap2.read()
        if not ret0 or not ret2: continue

        # Ректификация в оригинальном FullHD
        rect_full_l = cv2.remap(frame0, map_l_x, map_l_y, cv2.INTER_LINEAR)
        rect_full_r = cv2.remap(frame2, map_r_x, map_r_y, cv2.INTER_LINEAR)
        
        # На превью для UI жмем до 960x540, чтобы помещалось на экран
        rect_l_resized = cv2.resize(rect_full_l, (960, 540))
        cv2.imshow(window_name, rect_l_resized)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  
        if not is_frozen:
            print("\n[ЗАМОРОЗКА] Идет тяжелый просчет 3D сцены в FullHD...")
            
            gray_l = cv2.cvtColor(rect_full_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rect_full_r, cv2.COLOR_BGR2GRAY)
            
            # Программное выравнивание освещения
            mean_l, std_l = cv2.meanStdDev(gray_l)
            mean_r, std_r = cv2.meanStdDev(gray_r)
            gray_r = np.clip((gray_r.astype(np.float32) - mean_r) * (std_l / (std_r + 1e-5)) + mean_l, 0, 255).astype(np.uint8)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray_l = clahe.apply(gray_l)
            gray_r = clahe.apply(gray_r)

            # Карты диспаратности в полном разрешении
            disp_l = left_matcher.compute(gray_l, gray_r)
            disp_r = right_matcher.compute(gray_r, gray_l)
            filtered_disp = wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)
            
            # Проецируем строго по заводской Q
            points_3d = cv2.reprojectImageTo3D(filtered_disp.astype(np.float32) / 16.0, Q)
            
            frozen_image = rect_l_resized.copy()
            cv2.putText(frozen_image, "FROZEN. CLICK ANYWHERE!", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, frozen_image)
            is_frozen = True
            print("[ОК] Кадр зафиксирован.")
        else:
            print("[ЖИВОЙ ПОТОК] Возврат к видео.")
            is_frozen = False
            points_3d = None
            
    elif key == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
