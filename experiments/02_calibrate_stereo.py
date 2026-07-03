import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import glob
import sys

CHESSBOARD_SIZE = (9, 6)       # Внутренние углы
SQUARE_SIZE_MM = 26.0         # Размер квадрата

# Увеличим точность субпиксельного поиска
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

images_l = sorted(glob.glob('images/left/*.jpg'))
images_r = sorted(glob.glob('images/right/*.jpg'))

if len(images_l) == 0 or len(images_l) != len(images_r):
    print("Ошибка: Снимков нет или их количество не совпадает!")
    sys.exit(1)

print(f"Анализируем {len(images_l)} пар снимков...")

all_objpoints = []
all_imgpoints_l = []
all_imgpoints_r = []
valid_paths = []
img_shape = None

for img_l_path, img_r_path in zip(images_l, images_r):
    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)
    
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    
    if img_shape is None:
        img_shape = gray_l.shape[::-1]

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret_l and ret_r:
        corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        
        all_objpoints.append(objp)
        all_imgpoints_l.append(corners_l2)
        all_imgpoints_r.append(corners_r2)
        valid_paths.append((img_l_path, img_r_path))

if len(all_objpoints) == 0:
    print("Ошибка: Ни на одном кадре не удалось найти шахматную доску!")
    sys.exit(1)

# Этап 2: Одиночная калибровка для жесткой фильтрации
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(all_objpoints, all_imgpoints_l, img_shape, None, None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(all_objpoints, all_imgpoints_r, img_shape, None, None)

filtered_objpoints = []
filtered_imgpoints_l = []
filtered_imgpoints_r = []

for i in range(len(all_objpoints)):
    # Проверяем ЛЕВУЮ камеру
    imgpoints_l_projected, _ = cv2.projectPoints(all_objpoints[i], rvecs_l[i], tvecs_l[i], mtx_l, dist_l)
    error_l = cv2.norm(all_imgpoints_l[i], imgpoints_l_projected, cv2.NORM_L2) / len(imgpoints_l_projected)
    
    # Проверяем ПРАВУЮ камеру (ЭТОГО НЕ БЫЛО!)
    imgpoints_r_projected, _ = cv2.projectPoints(all_objpoints[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r)
    error_r = cv2.norm(all_imgpoints_r[i], imgpoints_r_projected, cv2.NORM_L2) / len(imgpoints_r_projected)
    
    # Отсекаем кадр, если ХОТЬ ОДНА камера выдала ошибку > 0.5 пикселя
    if error_l < 0.5 and error_r < 0.5:
        filtered_objpoints.append(all_objpoints[i])
        filtered_imgpoints_l.append(all_imgpoints_l[i])
        filtered_imgpoints_r.append(all_imgpoints_r[i])
    else:
        max_err = max(error_l, error_r)
        print(f"[-] Исключена плохая пара: {os.path.basename(valid_paths[i][0])} (Макс ошибка: {max_err:.2f} px)")

print(f"\nПосле жесткой фильтрации осталось пар: {len(filtered_objpoints)} из {len(all_objpoints)}")

if len(filtered_objpoints) < 8:
    print("Ошибка: Осталось слишком мало кадров!")
    sys.exit(1)

# Этап 3: Финальная стерео-калибровка со СТАБИЛЬНЫМИ флагами
print("Запуск финальной СТЕРЕО калибровки...")

# МЕНЯЕМ ФЛАГИ: 
# НЕ используем FIX_INTRINSIC. Вместо этого используем одиночные матрицы как начальное приближение (USE_INTRINSIC_GUESS)
# и разрешаем алгоритму их аккуратно донастроить под стереопару.
flags = (
    cv2.CALIB_USE_INTRINSIC_GUESS + 
    cv2.CALIB_SAME_FOCAL_LENGTH +     # Фокусное расстояние у одинаковых камер должно быть близким
    cv2.CALIB_ZERO_TANGENT_DIST       # Обнуляем тангенциальную дисторсию (она часто уводит алгоритм в ошибку)
)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    filtered_objpoints, filtered_imgpoints_l, filtered_imgpoints_r,
    mtx_l, dist_l, mtx_r, dist_r,
    img_shape, criteria=criteria, flags=flags
)

# Выравнивание
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
    img_shape, R, T, alpha=0
)

map_l_x, map_l_y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, img_shape, cv2.CV_32FC1)
map_r_x, map_r_y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, img_shape, cv2.CV_32FC1)

np.savez("stereo_calibration.npz", 
         map_l_x=map_l_x, map_l_y=map_l_y,
         map_r_x=map_r_x, map_r_y=map_r_y, Q=Q)

print(f"\n[+] УСПЕХ! Новая ошибка стерео-репроекции: {retval:.4f}")
