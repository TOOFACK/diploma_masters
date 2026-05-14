# NIR2 — Сегментация и векторизация городских облаков точек

Pipeline для семантической сегментации больших outdoor point clouds с последующей векторизацией в полигоны.
Объединяет 3D-сегментацию (Concerto / Point Transformer V3) и 2D BEV (SegFormer) и работает как на синтетических данных (SensatUrban), так и на реальных сканах.

## Содержание

- [Структура проекта](#структура-проекта)
- [Pipeline](#pipeline)
- [Установка](#установка)
- [Использование](#использование)
- [Результаты](#результаты)

## Структура проекта

```
NIR2/
├── Concerto/           # Concerto / PTv3 backbone + demo-скрипты инференса
├── Pointcept/          # Pointcept (используется для обучения PTv3)
├── SensatUrban_train/  # Подготовка данных и обучение на SensatUrban
├── iou_filter/         # Joint BEV+3D pipeline, merge-стратегии
├── mmsegmentation/     # SegFormer для BEV-ветки
├── vectorization/      # Векторизация масок → полигоны (новая часть)
├── data/               # Веса, датасеты, результаты инференса
```

## Pipeline

### Этап 1 — 3D-сегментация

Сырой PLY разбивается на тайлы 25×25 м, каждый прогоняется через **Concerto**
(`Concerto/demo/infer_sensaturban_scene.py`). Результаты сшиваются обратно в полное облако точек
с присвоенными классами (13 классов SensatUrban).

```bash
python Concerto/demo/infer_sensaturban_scene.py \
    --input scene.ply \
    --ckpt path/to/model_best.pth \
    --tile-size 25 --grid-size 0.05 \
    --estimate-normals \
    --save outputs/
```

### Этап 2 — Joint BEV + 3D (опционально)

Для повышения качества используется joint pipeline (`iou_filter/predict_ply_joint.py`):
параллельно 3D (Concerto) и 2D (SegFormer) предсказания, затем слияние.

**Стратегии слияния** (`iou_filter/merge_strategies.py`):
- `weighted_softmax` — α·softmax(BEV) + (1−α)·softmax(3D)
- `confidence_selection` — per-pixel выбор более уверенной модели
- `3d_priority` — BEV base, 3D перекрывает где confidence > 0.7

### Этап 3 — Проекция в BEV + сглаживание

```bash
python vectorization/ply_to_bev.py \
    --ply pred.ply --pred pred.npy \
    --out bev_mask.png \
    --resolution 0.2 --close-kernel 5 --merge-roads
```

Опции:
- `--resolution` — м/пиксель (0.05 — детально, 0.2 — плотнее для разреженных сканов)
- `--close-kernel` — морфологическое закрытие per-class для заливки щелей
- `--merge-roads` — Ground + Parking + Traffic Road → единый серый цвет

Дополнительное сглаживание перед векторизацией (Gaussian blur + fill holes + approxPolyDP):
```bash
python vectorization/smooth_bev_mask.py \
    --input bev_mask.png --config vectorization/input_format_real.yaml \
    --out bev_smoothed.png
```

### Этап 4 — Векторизация

```bash
python vectorization/vectorize_mask.py \
    --mask bev_smoothed.png \
    --config vectorization/input_format_real.yaml \
    --out_json result.json --out_png result.png \
    --color_space BGR
```

Per-class: извлечение бинарной маски по цвету → `cv2.findContours` → `cv2.approxPolyDP` (Douglas-Peucker).
Для дорог дополнительно применяется `road_processing.py` (skeletonize + ray casting для сшивки разрывов).

**Выход:** JSON c полигонами (`{points, area, anchor}`) и PNG превью с рендером по конфигу.

## Установка

Используется conda-окружение `pointcept-torch2.5.0-cu12.4`:

```bash
conda env create -f environment.yml  # PyTorch 2.5 + CUDA 12.4
# зависимости: open3d, opencv-python, scikit-image, shapely, PyYAML, matplotlib
```

Для инференса нужен GPU (~25 GB VRAM на тайл 25×25 м).

## Использование

### Полный pipeline на реальных данных

```bash
# 1. Инференс на GPU
python Concerto/demo/infer_sensaturban_scene.py \
    --input data/real_data/scene.ply \
    --ckpt data/weights/concerto_unfreeze_backbone_estimate_normals/model_best.pth \
    --tile-size 25 --grid-size 0.05 --estimate-normals \
    --save data/real_data/inference/

# 2. PLY → BEV
python vectorization/ply_to_bev.py \
    --ply data/real_data/inference/scene_pred.ply \
    --pred data/real_data/inference/scene_pred.npy \
    --out data/real_data/inference/bev.png \
    --resolution 0.2 --close-kernel 5 --merge-roads

# 3. Сглаживание
python vectorization/smooth_bev_mask.py \
    --input data/real_data/inference/bev.png \
    --config vectorization/input_format_real.yaml \
    --out data/real_data/inference/bev_smooth.png

# 4. Векторизация
python vectorization/vectorize_mask.py \
    --mask data/real_data/inference/bev_smooth.png \
    --config vectorization/input_format_real.yaml \
    --out_json data/real_data/inference/vectorized.json \
    --out_png data/real_data/inference/vectorized.png \
    --color_space BGR
```

### Визуализация

3D-рендер с вращением в GIF:
```bash
python vectorization/render_gif.py \
    --ply scene_pred.ply --out rotation.gif \
    --frames 300 --fps 20 --width 1600 --height 1600 \
    --elevation 40 --zoom 1.5 --point-size 2.5
```

## Результаты

### SensatUrban (birmingham_block_0)

| Стратегия | mIoU |
|---|---|
| only BEV | 46.3% |
| only 3D | 51.2% |
| 3D priority | 47.8% |
| confidence selection | 43.8% |
| weighted softmax | 49.3% |

### Поддерживаемые классы (SensatUrban, 13 классов)

Ground, Vegetation, Building, Wall, Bridge, Parking, Rail, Traffic Road,
Street Furniture, Car, Footpath, Bike, Water.
