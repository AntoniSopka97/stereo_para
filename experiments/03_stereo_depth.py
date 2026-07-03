import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import sys

# 1. Загружаем сохраненную калибровку
if not os.path.exists("stereo_calibration.npz"):
    print("Ошибка: Файл stereo_calibration.npz не найден! Сначала запусти скрипт калибровки.")
    sys.exit(1)

calib_data = np.load("stereo_calibration.npz")
map_l_x = calib_data["map_l_x"]
map_l_y = calib_data["map_l_y"]
map_r_x = calib_data["map_r_x"]
map_r_y = calib_data["map_r_y"]

# 2. Подключаем камеры к проверенным индексам
cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)

# Настраиваем MJPEG (разрешение берется максимальное, под него карты ректификации уже подстроены)
for cap in [cap0, cap2]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        

if not cap0.isOpened() or not cap2.isOpened():
    print("Ошибка открытия камер!")
    sys.exit(1)

# 3. Настройка детектора глубины (StereoBM)
# Эти параметры критически важны для дешевых камер.
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# Дополнительные фильтры для уменьшения шума
stereo.setMinDisparity(0)
stereo.setNumDisparities(64)       # Должно делиться на 16. Чем больше, тем ближе видит стереопара
stereo.setBlockSize(15)            # Размер окна сравнения (нечетное число от 5 до 255)
stereo.setTextureThreshold(10)     # Глушит плоские однотонные стены без текстуры
stereo.setUniquenessRatio(15)      # Фильтр ложных срабатываний
stereo.setSpeckleWindowSize(100)   # Удаляет мелкий "шум" и точки
stereo.setSpeckleRange(32)

window_name = "Real-Time Depth Map"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Создаем трекбары (ползунки) прямо в окне, чтобы ты мог крутить настройки в реальном времени!
def nothing(x): pass
cv2.createTrackbar("numDisparities", window_name, 4, 16, nothing) # Слайдер умножается на 16 в цикле
cv2.createTrackbar("blockSize", window_name, 7, 50, nothing)

print("\n[+] СТАРТ КАРТЫ ГЛУБИНЫ!")
print("Инструкция: Крути ползунки вверху окна, чтобы настроить четкость.")
print("Нажми 'q' для выхода.")

try:
    while True:
        ret0, frame0 = cap0.read()
        ret2, frame2 = cap2.read()
        
        if not ret0 or frame0 is None or not ret2 or frame2 is None:
            continue
            
        # Считываем настройки из ползунков
        num_disp = cv2.getTrackbarPos("numDisparities", window_name) * 16
        block_size = cv2.getTrackbarPos("blockSize", window_name) * 2 + 5 # Только нечетные >= 5
        
        if num_disp < 16: num_disp = 16
        stereo.setNumDisparities(num_disp)
        stereo.setBlockSize(block_size)

        # ШАГ А: Ректификация (выравнивание строк кадров по нашей геометрии)
        rectified_l = cv2.remap(frame0, map_l_x, map_l_y, cv2.INTER_LINEAR)
        rectified_r = cv2.remap(frame2, map_r_x, map_r_y, cv2.INTER_LINEAR)
        
        # ШАГ Б: Перевод в ЧБ (алгоритм StereoBM работает только с градациями серого)
        gray_l = cv2.cvtColor(rectified_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rectified_r, cv2.COLOR_BGR2GRAY)
        
        # ШАГ В: Расчет расхождения (Disparity)
        disparity = stereo.compute(gray_l, gray_r)
        
        # Нормализуем карту, чтобы перевести её в видимый диапазон 0-255
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, 
                                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # ШАГ Г: Раскрашиваем карту (близкие объекты будут яркими/теплыми, далекие — темными/холодными)
        depth_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
        
        # Для удобства склеиваем исходный левый выровненный кадр и карту глубины
        combined = cv2.hconcat([rectified_l, depth_color])
        
        cv2.imshow(window_name, combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nЗавершено пользователем")
finally:
    cap0.release()
    cap2.release()
    cv2.destroyAllWindows()
    print("Программа успешно завершена.")
