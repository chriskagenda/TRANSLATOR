"""Push README files to all 4 HuggingFace model repos."""
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "")
api = HfApi(token=token)

RELATED = """
## Related Models

| Model | Description |
|-------|-------------|
| [lunyoro-en2lun](https://huggingface.co/keithtwesigye/lunyoro-en2lun) | MarianMT English → Lunyoro |
| [lunyoro-lun2en](https://huggingface.co/keithtwesigye/lunyoro-lun2en) | MarianMT Lunyoro → English |
| [lunyoro-nllb_en2lun](https://huggingface.co/keithtwesigye/lunyoro-nllb_en2lun) | NLLB-200 English → Lunyoro |
| [lunyoro-nllb_lun2en](https://huggingface.co/keithtwesigye/lunyoro-nllb_lun2en) | NLLB-200 Lunyoro → English |

## Full Application

Source code: [chriskagenda/TRANSLATOR](https://github.com/chriskagenda/TRANSLATOR)
"""

DATASET_SECTION = """
## Dataset

The training data was compiled from:
- Crowd-sourced word and sentence submissions
- Runyoro-Rutooro dictionary entries (Excel)
- Parallel sentence corpora
- Back-translation augmentation (~53,948 total pairs, quality-filtered)
"""

readmes = {
    "keithtwesigye/lunyoro-en2lun": f"""---
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
{DATASET_SECTION}
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
{RELATED}""",

    "keithtwesigye/lunyoro-lun2en": f"""---
language:
- rw
- en
tags:
- translation
- lunyoro
- rutooro
- runyoro
- marianmt
- uganda
license: mit
---

# lunyoro-lun2en

Fine-tuned MarianMT model for **Lunyoro/Rutooro → English** translation.

Lunyoro-Rutooro is a Bantu language spoken by the Bunyoro-Kitara and Tooro kingdoms in western Uganda.

## Model Details

- Base model: `Helsinki-NLP/opus-mt-mul-en`
- Fine-tuned on: ~53,948 English-Lunyoro sentence pairs
- Training: 10 epochs, AdamW optimizer, cosine LR schedule
- Hardware: NVIDIA GPU (CUDA)
{DATASET_SECTION}
## Usage

```python
from transformers import MarianMTModel, MarianTokenizer

model_name = "keithtwesigye/lunyoro-lun2en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Oraire ota?"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
output = model.generate(**inputs, num_beams=4, max_length=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
{RELATED}""",

    "keithtwesigye/lunyoro-nllb_en2lun": f"""---
language:
- en
- rw
tags:
- translation
- lunyoro
- rutooro
- runyoro
- nllb
- nllb-200
- uganda
license: mit
---

# lunyoro-nllb_en2lun

Fine-tuned NLLB-200 model for **English → Lunyoro/Rutooro** translation.

Lunyoro-Rutooro is a Bantu language spoken by the Bunyoro-Kitara and Tooro kingdoms in western Uganda.

## Model Details

- Base model: `facebook/nllb-200-distilled-600M`
- Fine-tuned on: ~53,948 English-Lunyoro sentence pairs
- Training: 10 epochs, AdamW optimizer, cosine LR schedule
- Hardware: NVIDIA GPU (CUDA)
- Source language code: `eng_Latn`
- Target language code: `run_Latn`
{DATASET_SECTION}
## Usage

```python
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

model_name = "keithtwesigye/lunyoro-nllb_en2lun"
tokenizer = NllbTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.src_lang = "eng_Latn"
text = "How are you?"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
output = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("run_Latn"),
    num_beams=4,
    max_length=256
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
{RELATED}""",

    "keithtwesigye/lunyoro-nllb_lun2en": f"""---
language:
- rw
- en
tags:
- translation
- lunyoro
- rutooro
- runyoro
- nllb
- nllb-200
- uganda
license: mit
---

# lunyoro-nllb_lun2en

Fine-tuned NLLB-200 model for **Lunyoro/Rutooro → English** translation.

Lunyoro-Rutooro is a Bantu language spoken by the Bunyoro-Kitara and Tooro kingdoms in western Uganda.

## Model Details

- Base model: `facebook/nllb-200-distilled-600M`
- Fine-tuned on: ~53,948 English-Lunyoro sentence pairs
- Training: 10 epochs, AdamW optimizer, cosine LR schedule
- Hardware: NVIDIA GPU (CUDA)
- Source language code: `run_Latn`
- Target language code: `eng_Latn`
{DATASET_SECTION}
## Usage

```python
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

model_name = "keithtwesigye/lunyoro-nllb_lun2en"
tokenizer = NllbTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.src_lang = "run_Latn"
text = "Oraire ota?"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
output = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
    num_beams=4,
    max_length=256
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
{RELATED}""",
}

for repo_id, content in readmes.items():
    api.upload_file(
        path_or_fileobj=content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add README with model details and usage examples",
    )
    print(f"Done: {repo_id}")

print("\nAll READMEs pushed.")
