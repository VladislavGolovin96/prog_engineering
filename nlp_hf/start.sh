source ../venv/bin/activate
python main.py --text "Это было неожиданно круто!"
python main.py --text "ужасный сервис, больше не приду" --json
echo "Прекрасная погода и отличное настроение" > samples/sample.txt
python main.py --file samples/sample.txt