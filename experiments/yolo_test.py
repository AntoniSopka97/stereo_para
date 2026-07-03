import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import numpy as np
import sys
from ultralytics import YOLO

# ==========================================
# НАСТРОЙКА РАБОЧЕЙ ЗОНЫ (ЗАМЕРЬТЕ РУЛЕТКОЙ)
# Поставьте доску/банку на стол и впишите сюда расстояние от линз до стола в САНТИМЕТРАХ
DISTANCE_TO_BACKGROUND_CM = 211.8  # Например, 1 метр до плоскости
# ==========================================

if not os.path.exists("stereo_calibration.npz"):
    print("Ошибка: Нет файла stereo_calibration.npz!")
    sys.exit(1)

calib_data = np.load("stereo_calibration.npz")
map_l_x, map_l_y = calib_data["map_l_x"], calib_data["map_l_y"]
map_r_x, map_r_y = calib_data["map_r_x"], calib_data["map_r_y"]
Q = calib_data["Q"]

# Вытаскиваем честное фокусное расстояние (f) из матрицы калибровки Q
# В Q элемент [2, 3] или [3, 2] отвечает за f. Безопасный извлекатель:
f_pixel = abs(Q[2, 3])

# Если матрица Q была собрана иначе, вытащим фокус из элемента Q[1,1] или Q[0,0]
if f_pixel == 0 or f_pixel > 10000 or np.isinf(f_pixel) or np.isnan(f_pixel):
    f_pixel = abs(Q[1, 1])

print(f"[ИНФО] Успешно ИЗВЛЕЧЕНО реальное фокусное расстояние линз: {f_pixel:.2f} пикселей.")

print(f"[ИНФО] Успешно извлечено фокусное расстояние линз: {f_pixel:.1f} пикселей.")
print("[ИНФО] Загрузка нейросети YOLOv8...")
model = YOLO("yolov8n-seg.pt") 

cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)
for cap in [cap0, cap2]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

window_name = "YOLOv8 3D Industrial Measurer v3"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

print("\n--- СИСТЕМА СТАБИЛЬНОГО АВТОЗАМЕРА ЗАПУЩЕНА ---")
print(f"Положите объект на плоскость (расстояние до неё: {DISTANCE_TO_BACKGROUND_CM} см) и нажмите ПРОБЕЛ.")

is_frozen = False

while True:
    if not is_frozen:
        ret0, frame0 = cap0.read()
        ret2, frame2 = cap2.read()
        if not ret0 or not ret2: continue

        # Ректификация левого кадра
        rect_full_l = cv2.remap(frame0, map_l_x, map_l_y, cv2.INTER_LINEAR)
        cv2.imshow(window_name, cv2.resize(rect_full_l, (960, 540)))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if not is_frozen:
            print("\n[АНАЛИЗ] Распознавание объекта нейросетью...")
            results = model(rect_full_l, verbose=False)
            result = results[0]
            
            if result.boxes is None or len(result.boxes) == 0:
                print("[-] YOLO не увидела объектов. Измените ракурс.")
                continue
                
            # Ищем самый крупный объект в кадре (по площади описанной рамки)
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_box_idx = np.argmax(areas)
            
            x1, y1, x2, y2 = boxes[best_box_idx].astype(int)
            label = result.names[int(result.boxes.cls[best_box_idx])]
            
            print(f"[+] Распознан объект: '{label}'. Считаем физические габариты...")
            
            # Вычисляем размеры объекта в пикселях
            width_pixels = x2 - x1
            height_pixels = y2 - y1
            
            # --- МАТЕМАТИКА ОПОРНОЙ ПЛОСКОСТИ ---
            # Переводим заданное рулеткой расстояние Z в миллиметры
            z_mm = DISTANCE_TO_BACKGROUND_CM * 10.0
            
            # Формула оптического проецирования: Размер_мм = (Размер_пикс * Z_мм) / Фокус_пикс
            width_mm = (width_pixels * z_mm) / f_pixel
            height_mm = (height_pixels * z_mm) / f_pixel
            
            w_cm = width_mm / 10.0
            h_cm = height_mm / 10.0
            
            print(f"\n[РЕЗУЛЬТАТ АВТОЗАМЕРА ПО ПЛОСКОСТИ]:")
            print(f"-> Имя объекта: {label}")
            print(f"-> ФИЗИЧЕСКАЯ ШИРИНА: {w_cm:.2f} см")
            print(f"-> ФИЗИЧЕСКАЯ ДЛИНА/ВЫСОТА: {h_cm:.2f} см")
            
            # Отрисовка результатов на FullHD кадре
            display_frame = rect_full_l.copy()
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            # Подпись габаритов
            text_w = f"W: {w_cm:.1f} cm"
            text_h = f"H: {h_cm:.1f} cm"
            cv2.putText(display_frame, f"{label.upper()}", (x1 + 10, y1 - 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.putText(display_frame, text_w, (x1 + 10, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(display_frame, text_h, (x1 + 10, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            cv2.imshow(window_name, cv2.resize(display_frame, (960, 540)))
            is_frozen = True
        else:
            is_frozen = False

    elif key == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
