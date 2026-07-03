import os
# Глушим возможный спам логов
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import sys

print("Подключаемся к камерам 0 и 2...")
cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap1 = cv2.VideoCapture(2, cv2.CAP_V4L2)

# Настройка параметров для обеих камер
for cap in [cap0, cap1]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

if not cap0.isOpened() or not cap1.isOpened():
    print("Ошибка: Не удалось открыть одну из камер. Проверь индексы.")
    cap0.release()
    cap1.release()
    sys.exit(1)

# Создаем ОДНО окно для вывода обеих камер
window_name = "Stereo Pair Calibration"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

print("\nОкно запущено. Наведи камеры и выровняй их положение.")
print("Нажми 'q' для выхода.")

try:
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        # Если кадр временно пропустился — не падаем
        if not ret0 or frame0 is None or not ret1 or frame1 is None:
            continue
            
        # Добавляем текстовые метки на каждый кадр перед склейкой
        cv2.putText(frame0, "LEFT (Cam 0)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame1, "RIGHT (Cam 2)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Склеиваем кадры по горизонтали (горизонтальный конкатенированный массив)
        stereo_frame = cv2.hconcat([frame0, frame1])
        
        # Отображаем объединенную картинку в одном окне
        cv2.imshow(window_name, stereo_frame)
        
        # Ожидание нажатия 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nПрервано пользователем")

finally:
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    print("Ресурсы освобождены. Пока!")
