import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import sys
import numpy as np
import glob
import re

# Настройки доски для онлайн-проверки
CHESSBOARD_SIZE = (9, 6)

os.makedirs("images/left", exist_ok=True)
os.makedirs("images/right", exist_ok=True)

# --- НАЧАЛО БЛОКА АВТОПРОДОЛЖЕНИЯ ---
# Ищем все существующие файлы в левой папке, чтобы узнать последний номер
existing_files = glob.glob("images/left/img_*.jpg")
if existing_files:
    # Вытаскиваем числа из имен файлов (например, из 'images/left/img_31.jpg' достанет 31)
    indices = [int(re.search(r'img_(\d+)\.jpg', f).group(1)) for f in existing_files if re.search(r'img_(\d+)\.jpg', f)]
    if indices:
        saved_count = max(indices) + 1
    else:
        saved_count = 0
else:
    saved_count = 0

print(f"[ИНФО] Найдено прошлых кадров. Начинаем сбор со следующего индекса: {saved_count}")
# --- КОНЕЦ БЛОКА АВТОПРОДОЛЖЕНИЯ ---

cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)

for cap in [cap0, cap2]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

if not cap0.isOpened() or not cap2.isOpened():
    print("Ошибка: камеры не открылись!")
    sys.exit(1)

window_name = "Calibration Data Collector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
cv2.resizeWindow(window_name, 1280, 360) 

print("\n--- ИНСТРУКЦИЯ ---")
print("1. Старые кадры в безопасности! Продолжаем запись.")
print("2. Двигайте СТЕРЕОПАРУ камер к углам доски, делайте кадры БЛИЗКО и ДАЛЕКО.")
print("3. Нажмите ПРОБЕЛ, чтобы сделать снимок.")
print("4. Нажмите 'q' для завершения сбора.")

while True:
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()
    
    if not ret0 or not ret2:
        continue
        
    display_left = frame0.copy()
    display_right = frame2.copy()
    
    gray_l = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH)
    
    can_save = False
    
    if ret_l and ret_r:
        can_save = True
        cv2.drawChessboardCorners(display_left, CHESSBOARD_SIZE, corners_l, ret_l)
        cv2.drawChessboardCorners(display_right, CHESSBOARD_SIZE, corners_r, ret_r)
        cv2.circle(display_left, (50, 80), 15, (0, 255, 0), -1)
    else:
        cv2.circle(display_left, (50, 80), 15, (0, 0, 255), -1)

    cv2.putText(display_left, f"Saved: {saved_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    stereo_frame = cv2.hconcat([display_left, display_right])
    cv2.imshow(window_name, stereo_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '): 
        if can_save:
            cv2.imwrite(f"images/left/img_{saved_count:02d}.jpg", frame0)
            cv2.imwrite(f"images/right/img_{saved_count:02d}.jpg", frame2)
            print(f"[+] Снимок {saved_count} добавлен в общую базу!")
            saved_count += 1
        else:
            print("[-] Ошибка: Одна из камер (или обе) не видит доску. Снимок не сохранен!")
            
    elif key == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
print(f"Сбор окончен. Общее количество пар в папках: {saved_count}")
