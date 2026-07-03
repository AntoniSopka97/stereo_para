import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import numpy as np
import sys
import mediapipe as mp

# 1. Проверяем калибровку
if not os.path.exists("stereo_calibration.npz"):
    print("Ошибка: Сначала завершите калибровку!")
    sys.exit(1)

calib_data = np.load("stereo_calibration.npz")
map_l_x, map_l_y = calib_data["map_l_x"], calib_data["map_l_y"]
map_r_x, map_r_y = calib_data["map_r_x"], calib_data["map_r_y"]

# --- НАСТРОЙКА ГЛУБИНЫ (подставь свои параметры) ---
BASELINE = 17.5        # Расстояние между камерами в см
FOCAL_LENGTH = 357.9   # Фокусное расстояние из матрицы камеры

# Инициализация MediaPipe Hands (работает очень быстро на CPU)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Инициализация камер
cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)

for cap in [cap0, cap2]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Снизили разрешение до 720p для скорости
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

window_name = "Live Stereo Hand Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    t_start = cv2.getTickCount()
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()
    if not ret0 or not ret2: continue

    # Ректификация и ресайз до 640x360 (этого за глаза хватает для нейросети и экономит 400% ресурсов)
    rect_l = cv2.resize(cv2.remap(frame0, map_l_x, map_l_y, cv2.INTER_LINEAR), (640, 360))
    rect_r = cv2.resize(cv2.remap(frame2, map_r_x, map_r_y, cv2.INTER_LINEAR), (640, 360))

    # Переводим в RGB для MediaPipe
    rgb_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_l)

    # Если ладонь найдена
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Отрисовка скелета руки на левом кадре
            mp_draw.draw_landmarks(rect_l, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Берем точку основания ладони (точка 0) или центра (точка 9)
            # Координаты нормализованы от 0 до 1, переводим в пиксели
            h, w, _ = rect_l.shape
            x_l = int(hand_landmarks.landmark[9].x * w)
            y_l = int(hand_landmarks.landmark[9].y * h)
            
            # --- ПОИСК ТОЧКИ НА ПРАВОМ КАДРЕ (Шаблонный матчинг) ---
            # Вырезаем маленький квадрат вокруг точки на левом кадре
            patch_size = 16
            if (y_l - patch_size > 0 and y_l + patch_size < h and 
                x_l - patch_size > 0 and x_l + patch_size < w):
                
                template = cv2.cvtColor(rect_l[y_l-patch_size:y_l+patch_size, x_l-patch_size:x_l+patch_size], cv2.COLOR_BGR2GRAY)
                
                # Ищем этот квадрат на правой камере только вдоль той же строки (y_l)
                # Ограничиваем зону поиска по горизонтали (например, сдвиг до 120 пикселей)
                search_range = 120
                x_start = max(patch_size, x_l - search_range)
                strip = cv2.cvtColor(rect_r[y_l-patch_size:y_l+patch_size, x_start:x_l+patch_size], cv2.COLOR_BGR2GRAY)
                
                res = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                
                # Реальная координата X на правом кадре
                x_r = x_start + max_loc[0] + patch_size
                
                # Считаем сдвиг (диспарацию)
                disparity = abs(x_l - x_r)
                
                # Вычисляем расстояние в см
                if disparity > 1:
                    distance = (FOCAL_LENGTH * BASELINE) / disparity
                    
                    # Визуализируем точку и расстояние
                    cv2.circle(rect_l, (x_l, y_l), 7, (0, 0, 255), -1)
                    cv2.circle(rect_r, (x_r, y_l), 7, (0, 255, 0), -1)
                    
                    text = f"Distance: {distance:.1f} cm"
                    cv2.putText(rect_l, text, (x_l + 15, y_l), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Считаем FPS
    t_end = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (t_end - t_start)
    cv2.putText(rect_l, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Выводим оба кадра рядом
    combined = cv2.hconcat([rect_l, rect_r])
    cv2.imshow(window_name, combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
