import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import numpy as np
import sys

CHESSBOARD_SIZE = (9, 6)
REAL_SQUARE_SIZE_MM = 26.0

# НАСТРОЙКИ ФИЛЬТРА ЗЕЛЕНОЙ ЗОНЫ
MIN_Z_CUTOFF_MM = 450.0  # Игнорировать все, что ближе 45 см
MAX_Z_CUTOFF_MM = 2000.0 # Ограничим верхнюю планку двумя метрами для точности
EDGE_MARGIN_PERCENT = 0.12 # Отсекаем по 12% от краев кадра (зона дикой дисторсии линз)

if not os.path.exists("stereo_calibration.npz"):
    print("Ошибка: Нет файла stereo_calibration.npz!")
    sys.exit(1)

calib_data = np.load("stereo_calibration.npz")
map_l_x, map_l_y = calib_data["map_l_x"], calib_data["map_l_y"]
map_r_x, map_r_y = calib_data["map_r_x"], calib_data["map_r_y"]
Q = calib_data["Q"]

cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)
for cap in [cap0, cap2]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

window_name = "Stereo Size Validator v2"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

left_matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=256, blockSize=5,
    P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=2, uniquenessRatio=15,
    speckleWindowSize=100, speckleRange=2, mode=cv2.StereoSGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(10000.0)
wls_filter.setSigmaColor(1.5)

print("\n--- ЗАПУЩЕН МОДЕРНИЗИРОВАННЫЙ АНАЛИЗАТОР ---")
print("Жмите ПРОБЕЛ на разных расстояниях. Смотрите на вывод глубины Z.")

is_frozen = False

while True:
    if not is_frozen:
        ret0, frame0 = cap0.read()
        ret2, frame2 = cap2.read()
        if not ret0 or not ret2: continue

        rect_full_l = cv2.remap(frame0, map_l_x, map_l_y, cv2.INTER_LINEAR)
        rect_full_r = cv2.remap(frame2, map_r_x, map_r_y, cv2.INTER_LINEAR)
        
        rect_l_resized = cv2.resize(rect_full_l, (960, 540))
        cv2.imshow(window_name, rect_l_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if not is_frozen:
            print("\n[АНАЛИЗ] Просчет 3D и фильтрация...")
            
            gray_l = cv2.cvtColor(rect_full_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rect_full_r, cv2.COLOR_BGR2GRAY)
            H, W = gray_l.shape
            
            ret_corners, corners = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if not ret_corners:
                print("[-] Ошибка: Углы доски не найдены.")
                continue
                
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_2d = cv2.cornerSubPix(gray_l, corners, (11, 11), (-1, -1), criteria)
            
            # Выравнивание света
            mean_l, std_l = cv2.meanStdDev(gray_l)
            mean_r, std_r = cv2.meanStdDev(gray_r)
            gray_r = np.clip((gray_r.astype(np.float32) - mean_r) * (std_l / (std_r + 1e-5)) + mean_l, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray_l = clahe.apply(gray_l)
            gray_r = clahe.apply(gray_r)

            # 3D
            disp_l = left_matcher.compute(gray_l, gray_r)
            disp_r = right_matcher.compute(gray_r, gray_l)
            filtered_disp = wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)
            points_3d = cv2.reprojectImageTo3D(filtered_disp.astype(np.float32) / 16.0, Q)
            
            corners_grid = corners_2d.reshape((CHESSBOARD_SIZE[1], CHESSBOARD_SIZE[0], 2))
            
            measured_sizes = []
            z_distances = []
            display_frame = rect_full_l.copy()
            
            # Вычисляем границы отсечения краев в пикселях
            margin_x = int(W * EDGE_MARGIN_PERCENT)
            margin_y = int(H * EDGE_MARGIN_PERCENT)
            
            for r in range(CHESSBOARD_SIZE[1]):
                for c in range(CHESSBOARD_SIZE[0] - 1):
                    p1_2d = corners_grid[r, c].astype(int)
                    p2_2d = corners_grid[r, c+1].astype(int)
                    
                    # Проверка 1: Фильтр краев кадра
                    if (p1_2d[0] < margin_x or p1_2d[0] > W - margin_x or
                        p1_2d[1] < margin_y or p1_2d[1] > H - margin_y):
                        continue # Выкидываем крайние пиксели
                        
                    pt1_3d = points_3d[p1_2d[1], p1_2d[0]]
                    pt2_3d = points_3d[p2_2d[1], p2_2d[0]]
                    
                    if np.isinf(pt1_3d[2]) or np.isnan(pt1_3d[2]) or pt1_3d[2] <= 0:
                        continue
                        
                    # Проверка 2: Фильтр расстояния Z (Зеленая зона)
                    if not (MIN_Z_CUTOFF_MM < pt1_3d[2] < MAX_Z_CUTOFF_MM):
                        continue # Выкидываем слишком близкие/далекие замеры
                        
                    dist_3d = np.linalg.norm(pt1_3d - pt2_3d)
                    
                    if 15.0 < dist_3d < 45.0:
                        measured_sizes.append(dist_3d)
                        z_distances.append(pt1_3d[2])
                        cv2.line(display_frame, tuple(p1_2d), tuple(p2_2d), (0, 255, 0), 2)

            if len(measured_sizes) > 0:
                avg_size = np.mean(measured_sizes)
                avg_z_cm = np.mean(z_distances) / 10.0
                max_err = np.max(np.abs(np.array(measured_sizes) - REAL_SQUARE_SIZE_MM))
                
                print(f"\n[УСПЕХ] Валидных ребер после фильтрации: {len(measured_sizes)}")
                print(f"РАССТОЯНИЕ ДО ДОСКИ (Z): {avg_z_cm:.1f} см")
                print(f"Физический эталон: {REAL_SQUARE_SIZE_MM} мм")
                print(f"ЗАМЕР СТЕРЕОПАРОЙ: {avg_size:.2f} мм (Ошибка: {abs(avg_size-REAL_SQUARE_SIZE_MM):.2f} мм)")
                print(f"Макс. разброс на ребре: {max_err:.2f} мм")
                
                cv2.putText(display_frame, f"Z: {avg_z_cm:.1f}cm | Size: {avg_size:.2f}mm", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                print("[-] Кадр отфильтрован алгоритмом: доска находится в слепой зоне (слишком близко или на краю кадра)!")
                cv2.putText(display_frame, "OUT OF GREEN ZONE / SHUTDOWN", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            cv2.imshow(window_name, cv2.resize(display_frame, (960, 540)))
            is_frozen = True
        else:
            is_frozen = False

    elif key == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
