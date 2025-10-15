# Vision (PyTorch) классификация изображений ResNet-50

Скрипт определяет, что изображено на картинке, с помощью предобученной модели **ResNet-50** (`torchvision`), обученной на **ImageNet**.

---

## Установка 

```bash
# активируем виртуальное окружение из корня
source ../venv/bin/activate

# зависимости уже установлены из корневого requirements.txt
# (torch / torchvision / pillow и т.д.)

---

### Набор команд
# базовый запуск
python main.py --image samples/cat.jpg

# JSON-вывод
python main.py --image samples/cat.jpg --json

# больше результатов (Top-10)
python main.py --image samples/cat.jpg --topk 10

# сохранить копию изображения с подписью Top-1
python main.py --image samples/cat.jpg --save-vis out/cat_labeled.jpg

