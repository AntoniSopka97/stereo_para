import numpy as np

calib_data = np.load("stereo_calibration.npz")

# Проверим, какие матрицы у тебя сохранились. Обычно они называются M1 и M2 или K1 и K2
for key in calib_data.files:
    print(f"Найден массив: {key}")

# Попробуем вытащить фокусное расстояние (элемент [0,0] или [1,1] в матрице камеры)
# Если у тебя ключи называются иначе (например, K1 или cameraMatrix1), замени 'M1' на нужное имя
if "M1" in calib_data:
    matrix = calib_data["M1"]
    # Так как калибровка делалась на Full HD (1920x1080), а в коде для ладони 
    # мы жмем картинку в 3 раза (до 640x360), фокусное расстояние тоже нужно уменьшить в 3 раза
    fx_full_hd = matrix[0, 0]
    fx_for_script = fx_full_hd * (640 / 1920)
    print(f"\nТвоё точное FOCAL_LENGTH для скрипта с ладонью: {fx_for_script:.1f}")



import numpy as np

calib_data = np.load("stereo_calibration.npz")
Q = calib_data["Q"]

# Фокусное расстояние для Full HD лежит в матрице Q на позиции [2, 3]
fx_full_hd = Q[2, 3]

# Пересчитываем коэффициент уменьшения (1920 -> 640)
scale = 640 / 1920
fx_for_script = fx_full_hd * scale

print(f"Фокусное расстояние для Full HD: {fx_full_hd:.2f} px")
print(f"Вставь в скрипт с ладонью -> FOCAL_LENGTH = {abs(fx_for_script):.1f}")


