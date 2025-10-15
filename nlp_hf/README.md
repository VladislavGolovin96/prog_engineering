# NLP (Hugging Face) - анализ тональности текста

## Запуск
```bash
source ../venv/bin/activate
pip install -r ../requirements.txt -c ../constraints.txt
python main.py --text "Это было неожиданно круто!"
python main.py --text "ужасный сервис, больше не приду" --json
echo "Прекрасная погода и отличное настроение" > samples/sample.txt
python main.py --file samples/sample.txt
```
## Запуск скрипта с помощью sh !Перед запуском не забудьте навесить права запуска на файл

./start.sh