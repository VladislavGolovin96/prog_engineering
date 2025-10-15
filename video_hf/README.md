# Video (Hugging Face) — Object Detection with DETR

Обнаружение объектов на видео (по извлечённым кадрам) с помощью модели **facebook/detr-resnet-50**.

---

##  Установка 

```bash
# активируем виртуальное окружение из корня
source ../venv/bin/activate

# зависимости берём из корневых requirements.txt/constraints.txt
pip install -r ../requirements.txt -c ../constraints.txt

---
#Рекомендации к видео: 1080/720 fps: 15, время 5-15 сек. 

# базовый запуск: каждые 5 кадров, порог 0.5
python main.py --video samples/sample.mp4 --stride 5 --conf-thres 0.5

# сохранить превью первого обработанного кадра
python main.py --video samples/sample.mp4 --save-preview out/preview.jpg

# сохранить результат в CSV
python main.py --video samples/sample.mp4 --save-csv out/detections.csv

# JSON-вывод
python main.py --video samples/sample.mp4 --json
