source ../venv/bin/activate
python main.py --audio samples/car-horn.wav --topk 5
python main.py --audio samples/noise.wav --topk 5
python main.py --audio samples/voice.wav --topk 5
