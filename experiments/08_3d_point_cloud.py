import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import numpy as np
import sys
import open3d as o3d

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

window_name = "Stereo Live Preview (SPACE to view 3D Point Cloud)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

# Проверенные FullHD матчеры
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=256, blockSize=5,
    P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=2, uniquenessRatio=15,
    speckleWindowSize=100, speckleRange=2, mode=cv2.StereoSGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(10000.0)
wls_filter.setSigmaColor(1.5)

print("\n--- ИНСТРУКЦИЯ 3D ВИЗУАЛИЗАЦИИ ---")
print("1. Направьте камеры на себя, доску или комнату.")
print("2. Нажмите ПРОБЕЛ.")
print("3. Откроется отдельное окно Open3D с объемной моделью.")
print("4. Управление в 3D окне: Левая кнопка мыши — КРУТИТЬ, Колесико — МАСШТАБ, 'q' — Закрыть 3D.")

while True:
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()
    if not ret0 or not ret2: continue

    rect_full_l = cv2.remap(frame0, map_l_x, map_l_y, cv2.INTER_LINEAR)
    rect_full_r = cv2.remap(frame2, map_r_x, map_r_y, cv2.INTER_LINEAR)
    
    cv2.imshow(window_name, cv2.resize(rect_full_l, (960, 540)))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print("\n[МАГИЯ КИНО] Замораживаем кадр и строим 3D сцену...")
        
        gray_l = cv2.cvtColor(rect_full_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_full_r, cv2.COLOR_BGR2GRAY)
        
        # Программное выравнивание света
        mean_l, std_l = cv2.meanStdDev(gray_l)
        mean_r, std_r = cv2.meanStdDev(gray_r)
        gray_r = np.clip((gray_r.astype(np.float32) - mean_r) * (std_l / (std_r + 1e-5)) + mean_l, 0, 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray_l = clahe.apply(gray_l)
        gray_r = clahe.apply(gray_r)

        # Расчет диспаратности
        disp_l = left_matcher.compute(gray_l, gray_r)
        disp_r = right_matcher.compute(gray_r, gray_l)
        filtered_disp = wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)
        
        # Перепроецируем пиксели в честную геометрию XYZ (в мм)
        points_3d = cv2.reprojectImageTo3D(filtered_disp.astype(np.float32) / 16.0, Q)
        
        # Вытаскиваем цвета для каждой точки (OpenCV хранит BGR, а Open3D просит RGB во float от 0 до 1)
        colors_rgb = cv2.cvtColor(rect_full_l, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Вытягиваем матрицы в плоские массивы точек для построения облака
        flat_points = points_3d.reshape(-1, 3)
        flat_colors = colors_rgb.reshape(-1, 3)
        
        # --- ФИЛЬТР ШУМА И ДАЛЬНОСТИ ДЛЯ 3D ОКНА ---
        # Оставляем только точки, у которых координата Z (глубина) находится в диапазоне от 30 см до 2.5 метров
        z_coords = flat_points[:, 2]
        valid_mask = (~np.isnan(z_coords)) & (~np.isinf(z_coords)) & (z_coords > 300.0) & (z_coords < 2500.0)
        
        final_points = flat_points[valid_mask]
        final_colors = flat_colors[valid_mask]
        
        if len(final_points) == 0:
            print("[-] Ошибка: Нет валидных 3D точек в выбранном диапазоне дальности.")
            continue
            
        print(f"[+] Отрисовываем {len(final_points)} цветных точек в 3D пространстве...")
        
        # Создаем и наполняем объект Облака Точек Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_points)
        pcd.colors = o3d.utility.Vector3dVector(final_colors)
        
        # Бонус: Автоматическое удаление одиночных висящих в воздухе шумовых пикселей
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        
        # Инвертируем оси Y и Z, чтобы модель в окне не стояла вверх ногами относительно взгляда
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        # Запуск интерактивного просмотрщика
        o3d.visualization.draw_geometries([pcd], window_name="Open3D Point Cloud View", width=1024, height=768)
        print("[ЖИВОЙ ПОТОК] Возврат к видео. Нажмите ПРОБЕЛ для нового снимка.")
        
    elif key == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
