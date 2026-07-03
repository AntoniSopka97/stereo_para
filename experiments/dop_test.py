import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import numpy as np
import sys

# 1. Проверяем калибровку
if not os.path.exists("stereo_calibration.npz"):
    print("Ошибка: Сначала завершите калибровку!")
    sys.exit(1)

calib_data = np.load("stereo_calibration.npz")
map_l_x, map_l_y = calib_data["map_l_x"], calib_data["map_l_y"]
map_r_x, map_r_y = calib_data["map_r_x"], calib_data["map_r_y"]

# --- НАСТРОЙКА ДЛЯ РАСЧЕТА РАССТОЯНИЯ ---
# Открой свой файл калибровки или подставь сюда примерные значения:
# BASELINE = Расстояние между центрами линз камер в САНТИМЕТРАХ (например, 6.5 см)
# FOCAL_LENGTH = Возьми значение fx из матрицы камеры M1 (обычно около 600-1000)
BASELINE = 7.0  
FOCAL_LENGTH = 800.0 

# Переменные для хранения клика мыши
clicked_x, clicked_y = -1, -1
current_distance = "Кликните на карту глубины"

def mouse_callback(event, x, y, flags, param):
    global clicked_x, clicked_y
    if event == cv2.EVENT_LBUTTONDOWN:
        # Учитываем, что картинка склеена из двух частей (960 + 960)
        # Клик по правой половине (карте глубины) смещен на 960 пикселей
        if x >= 960:
            clicked_x = x - 960
            clicked_y = y

# Инициализация камер
cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)

for cap in [cap0, cap2]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

window_name = "Turbo Stereo Depth (Click for Distance)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)
cv2.setMouseCallback(window_name, mouse_callback)

def nothing(x): pass
cv2.createTrackbar('numDisparities', window_name, 6, 16, nothing) # Снизили макс. предел для скорости
cv2.createTrackbar('blockSize', window_name, 2, 8, nothing)

# Оптимизированный стерео-матчер (Быстрый MODE_SGBM вместо 3WAY)
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=96, blockSize=5,
    P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=2, mode=cv2.StereoSGBM_MODE_SGBM
)
# WLS фильтр умеет сам создавать правый матчер "под капотом", это быстрее на CPU!
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)

frame_counter = 0
num_disp = 96
b_size = 5

while True:
    t_start = cv2.getTickCount()
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()
    if not ret0 or not ret2: continue

    # Ректификация и быстрый ресайз
    rect_l = cv2.resize(cv2.remap(frame0, map_l_x, map_l_y, cv2.INTER_LINEAR), (960, 540))
    rect_r = cv2.resize(cv2.remap(frame2, map_r_x, map_r_y, cv2.INTER_LINEAR), (960, 540))

    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    # Программное выравнивание света
    mean_l, std_l = cv2.meanStdDev(gray_l)
    mean_r, std_r = cv2.meanStdDev(gray_r)
    gray_r = np.clip((gray_r.astype(np.float32) - mean_r[0][0]) * (std_l[0][0] / (std_r[0][0] + 1e-5)) + mean_l[0][0], 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_l = clahe.apply(gray_l)
    gray_r = clahe.apply(gray_r)

    # Опрашиваем трекбары раз в 15 кадров, чтобы не тормозить UI поток
    frame_counter += 1
    if frame_counter % 15 == 0:
        num_disp = cv2.getTrackbarPos('numDisparities', window_name) * 16
        if num_disp == 0: num_disp = 16
        b_size = cv2.getTrackbarPos('blockSize', window_name) * 2 + 3
        
        left_matcher.setNumDisparities(num_disp)
        left_matcher.setBlockSize(b_size)
        left_matcher.setP1(8 * 3 * b_size ** 2)
        left_matcher.setP2(32 * 3 * b_size ** 2)

    # Быстрый двойной расчет силами OpenCV C++ (без вызова right_matcher в Python)
    disp_l = left_matcher.compute(gray_l, gray_r)
    # Используем встроенный генератор правой карты
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    disp_r = right_matcher.compute(gray_r, gray_l)

    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    filtered_disp = wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)

    # --- РАСЧЕТ РАССТОЯНИЯ ДЛЯ ВЫБРАННОЙ ТОЧКИ ---
    if clicked_x != -1 and clicked_y != -1:
        # Получаем значение диспарации (сдвига) в выбранном пикселе.
        # Делим на 16.0, так как OpenCV хранит диспарацию в фиксированной точке (умноженной на 16)
        disparity_value = filtered_disp[clicked_y, clicked_x] / 16.0
        
        if disparity_value > 0:
            # Формула триангуляции: Z = (f * B) / d
            distance_cm = (FOCAL_LENGTH * BASELINE) / disparity_value
            if distance_cm > 100:
                current_distance = f"Dist: {distance_cm/100:.2f} m"
            else:
                current_distance = f"Dist: {distance_cm:.1f} cm"
        else:
            current_distance = "Dist: Undefined (Hole)"

    # Визуализация
    disp_vis = cv2.ximgproc.getDisparityVis(filtered_disp, None, 1.0)
    
    # Морфологическое закрытие для замазывания шума на карте
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    disp_vis = cv2.morphologyEx(disp_vis, cv2.MORPH_CLOSE, kernel)
    
    depth_colormap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA)
    depth_colormap[disp_vis <= 5] = (0, 0, 0)

    # Рисуем прицел на выбранной точке
    if clicked_x != -1 and clicked_y != -1:
        cv2.circle(depth_colormap, (clicked_x, clicked_y), 5, (0, 255, 0), -1)
        cv2.putText(depth_colormap, current_distance, (clicked_x + 10, clicked_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Считаем чистый FPS
    t_end = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (t_end - t_start)
    cv2.putText(rect_l, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    combined = cv2.hconcat([rect_l, depth_colormap])
    cv2.imshow(window_name, combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
