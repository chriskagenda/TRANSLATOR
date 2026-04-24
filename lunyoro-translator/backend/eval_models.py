"""
Evaluate all 4 models on the test set.
Metrics: BLEU, token F1, exact match
Usage: python eval_models.py
"""
import json, time
from pathlib import Path
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer, NllbTokenizer, AutoModelForSeq2SeqLM

DATA_DIR  = Path(__file__).parent / "data" / "training"
MODEL_DIR = Path(__file__).parent / "model"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH     = 16


def bleu_score(refs, hyps):
    from sacrebleu.metrics import BLEU
    return BLEU(effective_order=True).corpus_score(hyps, [refs]).score


def token_f1(ref: str, hyp: str) -> float:
    r, h = set(ref.lower().split()), set(hyp.lower().split())
    if not r or not h:
        return 0.0
    p = len(r & h) / len(h)
    rec = len(r & h) / len(r)
    return 2 * p * rec / (p + rec) if (p + rec) else 0.0


def batch_translate_marian(texts, model, tokenizer, max_len=256):
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, num_beams=8, max_length=max_len)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in out]


def batch_translate_nllb(texts, model, tokenizer, src_lang, tgt_lang, max_len=256):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len).to(DEVICE)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=tgt_id,
                             num_beams=8, max_length=max_len)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in out]


def evaluate(name, translate_fn, src_col, tgt_col, df):
    srcs = df[src_col].tolist()
    refs = df[tgt_col].tolist()
    hyps = []
    t0 = time.time()
    for i in range(0, len(srcs), BATCH):
        hyps += translate_fn(srcs[i:i+BATCH])
    elapsed = (time.time() - t0) * 1000 / len(srcs)

    bleu  = bleu_score(refs, hyps)
    f1    = sum(token_f1(r, h) for r, h in zip(refs, hyps)) / len(refs) * 100
    exact = sum(r.strip().lower() == h.strip().lower() for r, h in zip(refs, hyps)) / len(refs) * 100

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  samples : {len(srcs)}")
    print(f"  BLEU    : {bleu:.2f}")
    print(f"  Token F1: {f1:.2f}%")
    print(f"  Exact   : {exact:.2f}%")
    print(f"  ms/sample: {elapsed:.1f}")
    return {"model": name, "samples": len(srcs), "bleu": round(bleu,2),
            "token_f1": round(f1,2), "exact_match_pct": round(exact,2),
            "ms_per_sample": round(elapsed,1)}


def main():
    test = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=["english","lunyoro"])
    print(f"Test set: {len(test)} pairs  |  device: {DEVICE}")
    results = []

    # --- MarianMT en2lun ---
    print("\nLoading MarianMT en2lun...")
    tok = MarianTokenizer.from_pretrained(str(MODEL_DIR / "en2lun"))
    mdl = MarianMTModel.from_pretrained(str(MODEL_DIR / "en2lun")).to(DEVICE).eval()
    results.append(evaluate("MarianMT en→lun",
        lambda t: batch_translate_marian(t, mdl, tok),
        "english", "lunyoro", test))
    del mdl

    # --- MarianMT lun2en ---
    print("\nLoading MarianMT lun2en...")
    tok = MarianTokenizer.from_pretrained(str(MODEL_DIR / "lun2en"))
    mdl = MarianMTModel.from_pretrained(str(MODEL_DIR / "lun2en")).to(DEVICE).eval()
    results.append(evaluate("MarianMT lun→en",
        lambda t: batch_translate_marian(t, mdl, tok),
        "lunyoro", "english", test))
    del mdl

    # --- NLLB en2lun ---
    print("\nLoading NLLB en2lun...")
    tok = NllbTokenizer.from_pretrained(str(MODEL_DIR / "nllb_en2lun"))
    mdl = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR / "nllb_en2lun")).to(DEVICE).eval()
    results.append(evaluate("NLLB-200 en→lun",
        lambda t: batch_translate_nllb(t, mdl, tok, "eng_Latn", "run_Latn"),
        "english", "lunyoro", test))
    del mdl

    # --- NLLB lun2en ---
    print("\nLoading NLLB lun2en...")
    tok = NllbTokenizer.from_pretrained(str(MODEL_DIR / "nllb_lun2en"))
    mdl = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR / "nllb_lun2en")).to(DEVICE).eval()
    results.append(evaluate("NLLB-200 lun→en",
        lambda t: batch_translate_nllb(t, mdl, tok, "run_Latn", "eng_Latn"),
        "lunyoro", "english", test))
    del mdl

    out = Path(__file__).parent / "eval_results_full.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
