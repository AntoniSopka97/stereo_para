import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import numpy as np
import glob
import sys

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 26.0
MIN_FRAMES_TO_KEEP = 15  
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

def compute_stereo_score(images_l, images_r):
    all_objpoints, all_imgpoints_l, all_imgpoints_r = [], [], []
    img_shape = None

    for img_l_path, img_r_path in zip(images_l, images_r):
        img_l = cv2.imread(img_l_path)
        img_r = cv2.imread(img_r_path)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = gray_l.shape[::-1]

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, None)

        if ret_l and ret_r:
            corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            all_objpoints.append(objp)
            all_imgpoints_l.append(corners_l2)
            all_imgpoints_r.append(corners_r2)

    if len(all_objpoints) < 5:
        return 999, None, None, None, None, None

    _, mtx_l, dist_l, _, _ = cv2.calibrateCamera(all_objpoints, all_imgpoints_l, img_shape, None, None)
    _, mtx_r, dist_r, _, _ = cv2.calibrateCamera(all_objpoints, all_imgpoints_r, img_shape, None, None)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST
    try:
        retval, _, _, _, _, _, _, _, _ = cv2.stereoCalibrate(
            all_objpoints, all_imgpoints_l, all_imgpoints_r,
            mtx_l, dist_l, mtx_r, dist_r, img_shape, criteria=criteria, flags=flags
        )
        return retval, all_objpoints, all_imgpoints_l, all_imgpoints_r, mtx_l, dist_l
    except:
        return 999, None, None, None, None, None

print("=== ЗАПУСК АВТОМАТИЧЕСКОГО КИЛЛЕРА КАДРОВ ===")

while True:
    images_l = sorted(glob.glob('images/left/*.jpg'))
    images_r = sorted(glob.glob('images/right/*.jpg'))
    current_count = len(images_l)
    
    if current_count <= MIN_FRAMES_TO_KEEP:
        print(f"\n[!] Стоп: Достигнут лимит минимального количества кадров ({MIN_FRAMES_TO_KEEP}).")
        break
        
    base_err, all_obj, all_img_l, all_img_r, mtx_l, dist_l = compute_stereo_score(images_l, images_r)
    print(f"\nРазмер базы: {current_count} кадров | Текущая ошибка: {base_err:.4f} px")
    print("Идет обсчет вариантов комбинаций...")
    
    worst_file_idx = -1
    max_improvement = 0.001 
    
    # Ищем худший кадр с выводом прогресса в реальном времени
    for i in range(current_count):
        # Выводим динамическую строку прогресса \r перезаписывает текущую строку консоли
        sys.stdout.write(f"\r  Прогресс итерации: [{i+1}/{current_count}] Проверяем исключение: {os.path.basename(images_l[i])} ...")
        sys.stdout.flush()
        
        sub_l = images_l[:i] + images_l[i+1:]
        sub_r = images_r[:i] + images_r[i+1:]
        
        err, _, _, _, _, _ = compute_stereo_score(sub_l, sub_r)
        improvement = base_err - err
        
        if improvement > max_improvement:
            max_improvement = improvement
            worst_file_idx = i

    # Очищаем строку прогресса перед выводом результата шага
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    if worst_file_idx != -1:
        bad_l = images_l[worst_file_idx]
        bad_r = images_r[worst_file_idx]
        
        print(f"  [-] Удален вредитель: {os.path.basename(bad_l)} (Ошибка упадет на: +{max_improvement:.4f} px)")
        
        if os.path.exists(bad_l): os.remove(bad_l)
        if os.path.exists(bad_r): os.remove(bad_r)
    else:
        print("[+] Оптимизация завершена! Ошибка стабилизировалась, плохих кадров больше нет.")
        break

final_err, _, _, _, _, _ = compute_stereo_score(sorted(glob.glob('images/left/*.jpg')), sorted(glob.glob('images/right/*.jpg')))
print(f"\n=== ИТОГ: Осталось идеальных кадров: {len(glob.glob('images/left/*.jpg'))} | Финальная ошибка: {final_err:.4f} px ===")
