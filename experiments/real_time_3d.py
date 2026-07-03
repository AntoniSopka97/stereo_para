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

# Масштабируем Q под разрешение 480x270 (уменьшение в 4 раза от 1920x1080)
# Делим строго фокусное расстояние и оптические центры
scale_factor = 4.0
Q_live = Q.copy()
Q_live[0, 3] /= scale_factor  # cx
Q_live[1, 3] /= scale_factor  # cy
Q_live[2, 3] /= scale_factor  # f
Q_live[3, 2] *= scale_factor  # 1/Baseline

cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)
for cap in [cap0, cap2]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Быстрый матчер (меньше numDisparities и blockSize для дикой скорости)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64, # Сжатая картинка = меньше сдвиг пикселей
    blockSize=5,
    P1=8*3*5**2,
    P2=32*3*5**2,
    disp12MaxDiff=2,
    uniquenessRatio=10,
    mode=cv2.StereoSGBM_MODE_SGBM_3WAY
)

# Инициализируем живое окно Open3D
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Live 3D Point Cloud", width=1024, height=768)

pcd = o3d.geometry.PointCloud()
is_first_frame = True

print("\n[ОК] Визуализатор запущен. Крутите сцену прямо во время стрима! Нажмите ESC для выхода.")

while True:
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()
    if not ret0 or not ret2: continue

    # Ректификация и жесткий даунсемплинг для ультра-скорости
    rect_full_l = cv2.remap(frame0, map_l_x, map_l_y, cv2.INTER_LINEAR)
    rect_full_r = cv2.remap(frame2, map_r_x, map_r_y, cv2.INTER_LINEAR)
    
    small_l = cv2.resize(rect_full_l, (480, 270), interpolation=cv2.INTER_AREA)
    small_r = cv2.resize(rect_full_r, (480, 270), interpolation=cv2.INTER_AREA)

    gray_l = cv2.cvtColor(small_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(small_r, cv2.COLOR_BGR2GRAY)

    # Быстрый обсчет стерео (без WLS фильтра, так как он сильно тормозит поток)
    disp = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    
    # 3D точки и цвета
    points_3d = cv2.reprojectImageTo3D(disp, Q_live).reshape(-1, 3)
    colors_rgb = cv2.cvtColor(small_l, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32) / 255.0

    # Фильтр дистанции (теперь расширим до 4.5 метров, как вы просили)
    z_coords = points_3d[:, 2]
    valid_mask = (~np.isnan(z_coords)) & (~np.isinf(z_coords)) & (z_coords > 300.0) & (z_coords < 4500.0)
    
    if not np.any(valid_mask): continue

    # Обновляем геометрию облака точек на лету
    pcd.points = o3d.utility.Vector3dVector(points_3d[valid_mask])
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb[valid_mask])
    
    # Переворачиваем, чтобы картинка не была зеркальной/перевернутой
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if is_first_frame:
        vis.add_geometry(pcd)
        is_first_frame = False
    else:
        vis.update_geometry(pcd)
        
    vis.poll_events()
    vis.update_renderer()

    # Обычный костыль для выхода из бесконечного цикла по ESC
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap0.release()
cap2.release()
vis.destroy_window()
