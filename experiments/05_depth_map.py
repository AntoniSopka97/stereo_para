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

# 2. Инициализация камер
cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)

for cap in [cap0, cap2]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

window_name = "Filtered Stereo Depth Map"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

def nothing(x): pass
# Увеличим дефолтное значение numDisparities для работы на близком расстоянии
cv2.createTrackbar('numDisparities', window_name, 12, 24, nothing) 
cv2.createTrackbar('blockSize', window_name, 2, 10, nothing)
cv2.createTrackbar('wls_lambda', window_name, 80, 150, nothing)

# Базовый стерео-матчер
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=192, blockSize=5,
    P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=2, uniquenessRatio=15,
    speckleWindowSize=100, speckleRange=2, mode=cv2.StereoSGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)

print("\n[ОК] Скрипт запущен. Яркость выравнивается программно.")

while True:
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()
    if not ret0 or not ret2: continue

    # Ректификация и ресайз до 960x540
    rect_l = cv2.resize(cv2.remap(frame0, map_l_x, map_l_y, cv2.INTER_LINEAR), (960, 540))
    rect_r = cv2.resize(cv2.remap(frame2, map_r_x, map_r_y, cv2.INTER_LINEAR), (960, 540))

    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    # --- НАЧАЛО ПРОГРАММНОГО ВЫРАВНИВАНИЯ ЯРКОСТИ И КОНТРАСТА ---
    # Считаем среднее и отклонение для левого кадра
    mean_l, std_l = cv2.meanStdDev(gray_l)
    # Считаем среднее и отклонение для правого кадра
    mean_r, std_r = cv2.meanStdDev(gray_r)
    
    # Линейно трансформируем правый кадр под яркость левого кадра
    # Формула: New_R = (R - Mean_R) * (Std_L / Std_R) + Mean_L
    gray_r = np.clip((gray_r.astype(np.float32) - mean_r[0][0]) * (std_l[0][0] / (std_r[0][0] + 1e-5)) + mean_l[0][0], 0, 255).astype(np.uint8)

    # Накатываем адаптивный контраст, чтобы проявить текстуру досок
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_l = clahe.apply(gray_l)
    gray_r = clahe.apply(gray_r)
    # --- КОНЕЦ ВЫРАВНИВАНИЯ (теперь картинки идеально идентичны по свету) ---

    num_disp = cv2.getTrackbarPos('numDisparities', window_name) * 16
    if num_disp == 0: num_disp = 16
    b_size = cv2.getTrackbarPos('blockSize', window_name) * 2 + 3
    wls_lam = cv2.getTrackbarPos('wls_lambda', window_name) * 100

    left_matcher.setNumDisparities(num_disp)
    left_matcher.setBlockSize(b_size)
    left_matcher.setP1(8 * 1 * b_size ** 2)
    left_matcher.setP2(32 * 1 * b_size ** 2)
    
    right_matcher.setNumDisparities(num_disp)
    right_matcher.setBlockSize(b_size)

    wls_filter.setLambda(float(wls_lam))
    wls_filter.setSigmaColor(1.5)

    disp_l = left_matcher.compute(gray_l, gray_r)
    disp_r = right_matcher.compute(gray_r, gray_l)

    filtered_disp = wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)

    disp_vis = cv2.ximgproc.getDisparityVis(filtered_disp, None, 1.0)
    depth_colormap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # Гасим совсем плохие точки в черный
    depth_colormap[disp_vis <= 5] = (0, 0, 0)

    combined = cv2.hconcat([rect_l, depth_colormap])
    cv2.imshow(window_name, combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
