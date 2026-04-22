# Extra Training Data

Drop any additional CSV datasets here to include them in training.

## Required format

Each CSV must have exactly these two column names:

```
english,lunyoro
"Hello how are you","Oraire ota"
"I am fine","Ndi bulamu"
"What is your name","Eizina ryawe ni rani"
```

## How to retrain after adding data

```bash
python prepare_training_data.py   # rebuilds training splits
python fine_tune.py --direction both --epochs 10 --batch_size 32
```

## Notes
- Column names are case-insensitive (English, ENGLISH, english all work)
- Duplicate pairs are automatically removed
- Files are picked up automatically — no code changes needed
