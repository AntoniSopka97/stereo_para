import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import sys
import glob
import re
import time

os.makedirs("images/left", exist_ok=True)
os.makedirs("images/right", exist_ok=True)

# --- АВТОПРОДОЛЖЕНИЕ БАЗЫ ---
existing_files = glob.glob("images/left/img_*.jpg")
if existing_files:
    indices = [int(re.search(r'img_(\d+)\.jpg', f).group(1)) for f in existing_files if re.search(r'img_(\d+)\.jpg', f)]
    saved_count = max(indices) + 1 if indices else 0
else:
    saved_count = 0

print(f"[ИНФО] База сохранена. Начинаем автосбор со снимка №: {saved_count}")

cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)

# Выставляем FullHD
for cap in [cap0, cap2]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

if not cap0.isOpened() or not cap2.isOpened():
    print("Ошибка: камеры не открылись!")
    sys.exit(1)

window_name = "Fast Data Collector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
# Делаем маленькое превью окно для разгрузки видеопамяти
cv2.resizeWindow(window_name, 640, 180) 

print("\n--- СКОРОСТНОЙ АВТОСБОР (0.5 сек) ---")
print("1. Просто плавно вози доской перед камерами.")
print("2. Снимки делаются автоматически каждые 0.5 сек.")
print("3. Смазанные кадры вырежет скрипт фильтрации.")
print("4. Нажми 'q' для выхода.")

last_save_time = time.time()
interval = 0.5  # Жесткая задержка в полсекунды

try:
    while True:
        ret0, frame0 = cap0.read()
        ret2, frame2 = cap2.read()
        
        if not ret0 or frame0 is None or not ret2 or frame2 is None:
            continue
            
        current_time = time.time()
        
        # Сжатое превью для экрана пользователя (оригиналы frame0/frame2 остаются чистыми в FullHD)
        preview_l = cv2.resize(frame0, (640, 360))
        preview_r = cv2.resize(frame2, (640, 360))
        
        # Срабатывание по таймеру
        if current_time - last_save_time >= interval:
            # Сохраняем оригинальные FullHD кадры без сжатия
            cv2.imwrite(f"images/left/img_{saved_count:02d}.jpg", frame0)
            cv2.imwrite(f"images/right/img_{saved_count:02d}.jpg", frame2)
            print(f"[АВТО] Сохранен FullHD снимок {saved_count}")
            saved_count += 1
            last_save_time = current_time
            
            # Индикация вспышки на маленьком превью
            cv2.rectangle(preview_l, (0, 0), (640, 360), (0, 255, 0), 10)

        cv2.putText(preview_l, f"Saved: {saved_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Склеиваем маленькие превьюшки для вывода на экран
        stereo_preview = cv2.hconcat([preview_l, preview_r])
        cv2.imshow(window_name, stereo_preview)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nПрервано пользователем")
finally:
    cap0.release()
    cap2.release()
    cv2.destroyAllWindows()
    print(f"Сбор завершен. Всего снимков в базе: {saved_count}")
