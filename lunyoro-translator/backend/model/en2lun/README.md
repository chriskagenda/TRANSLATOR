---
language:
- en
- rw
tags:
- translation
- lunyoro
- rutooro
- runyoro
- marianmt
- uganda
license: mit
---

# lunyoro-en2lun

Fine-tuned MarianMT model for **English → Lunyoro/Rutooro** translation.

Lunyoro-Rutooro is a Bantu language spoken by the Bunyoro-Kitara and Tooro kingdoms in western Uganda.

## Model Details

- Base model: `Helsinki-NLP/opus-mt-en-mul`
- Fine-tuned on: ~53,948 English-Lunyoro sentence pairs
- Training: 10 epochs, AdamW optimizer, cosine LR schedule
- Hardware: NVIDIA GPU (CUDA)

## Dataset

The training data was compiled from:
- Crowd-sourced word and sentence submissions
- Runyoro-Rutooro dictionary entries (Excel)
- Parallel sentence corpora
- Back-translation augmentation (~53,948 total pairs, quality-filtered)

## Usage

```python
from transformers import MarianMTModel, MarianTokenizer

model_name = "keithtwesigye/lunyoro-en2lun"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "How are you?"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
output = model.generate(**inputs, num_beams=4, max_length=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Related Models

| Model | Description |
|-------|-------------|
| [lunyoro-en2lun](https://huggingface.co/keithtwesigye/lunyoro-en2lun) | MarianMT English → Lunyoro |
| [lunyoro-lun2en](https://huggingface.co/keithtwesigye/lunyoro-lun2en) | MarianMT Lunyoro → English |
| [lunyoro-nllb_en2lun](https://huggingface.co/keithtwesigye/lunyoro-nllb_en2lun) | NLLB-200 English → Lunyoro |
| [lunyoro-nllb_lun2en](https://huggingface.co/keithtwesigye/lunyoro-nllb_lun2en) | NLLB-200 Lunyoro → English |

## Full Application

Source code: [chriskagenda/TRANSLATOR](https://github.com/chriskagenda/TRANSLATOR)
