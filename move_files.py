import os
import shutil
from glob import glob
import random

# === Укажи путь, где лежат ВСЕ скачанные sensat *.zip распаковки ===
INPUT_ROOT = "/home/pavel/ITMO/NIR2/data/SensetUrban"       # <- меняешь на свой путь
OUTPUT_ROOT = "/home/pavel/ITMO/NIR2/data/SensetUrban_converted"          # <- куда собрать dataset

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "test"), exist_ok=True)

print(">>> Сканирую все PLY внутри:", INPUT_ROOT)
all_ply = glob(os.path.join(INPUT_ROOT, "**/*.ply"), recursive=True)
print("Найдено PLY:", len(all_ply))

# --- распределение train/test по содержимому пути ---
for ply_file in all_ply:
    name = os.path.basename(ply_file)

    # внутри путей иногда есть /train/... или /test/...
    if "/train/" in ply_file or "\\train\\" in ply_file:
        dst = os.path.join(OUTPUT_ROOT, "train", name)
    elif "/test/" in ply_file or "\\test\\" in ply_file:
        dst = os.path.join(OUTPUT_ROOT, "test", name)
    else:
        # если файл просто лежит в корне или странной папке — добавим в train
        dst = os.path.join(OUTPUT_ROOT, "train", name)

    shutil.copy2(ply_file, dst)

print("Готово. Train:", len(os.listdir(os.path.join(OUTPUT_ROOT, "train"))),
      "Test:", len(os.listdir(os.path.join(OUTPUT_ROOT, "test"))))

# === Теперь делаем train/val split (10% на валидацию) ===

train_files = sorted(os.listdir(os.path.join(OUTPUT_ROOT, "train")))
random.shuffle(train_files)

val_size = max(1, int(len(train_files) * 0.1))
val_split = train_files[:val_size]
train_split = train_files[val_size:]

print("Train:", len(train_split), "Val:", len(val_split))

# === Записываем список файлов ===

with open(os.path.join(OUTPUT_ROOT, "train_list.txt"), "w") as f:
    for name in train_split:
        f.write(f"train/{name}\n")

with open(os.path.join(OUTPUT_ROOT, "val_list.txt"), "w") as f:
    for name in val_split:
        f.write(f"train/{name}\n")

print(">>> Файлы train_list.txt и val_list.txt созданы.")
print(">>> Датасет полностью готов.")
