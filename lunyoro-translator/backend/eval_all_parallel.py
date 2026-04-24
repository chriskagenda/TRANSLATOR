"""
Evaluate all 4 models in parallel across 2 GPUs.
  GPU 0: MarianMT en2lun + lun2en
  GPU 1: NLLB en2lun + lun2en
"""
import json, time, multiprocessing as mp
from pathlib import Path
import pandas as pd
import torch

DATA_DIR  = Path(__file__).parent / "data" / "training"
MODEL_DIR = Path(__file__).parent / "model"
BATCH     = 32


def bleu_score(refs, hyps):
    from sacrebleu.metrics import BLEU
    return BLEU(effective_order=True).corpus_score(hyps, [refs]).score


def token_f1(ref: str, hyp: str) -> float:
    r, h = set(ref.lower().split()), set(hyp.lower().split())
    if not r or not h: return 0.0
    p = len(r & h) / len(h)
    rec = len(r & h) / len(r)
    return 2 * p * rec / (p + rec) if (p + rec) else 0.0


def compute_metrics(refs, hyps, name, elapsed_ms):
    bleu  = bleu_score(refs, hyps)
    f1    = sum(token_f1(r, h) for r, h in zip(refs, hyps)) / len(refs) * 100
    exact = sum(r.strip().lower() == h.strip().lower() for r, h in zip(refs, hyps)) / len(refs) * 100
    result = {
        "model": name, "samples": len(refs),
        "bleu": round(bleu, 2), "token_f1": round(f1, 2),
        "exact_match_pct": round(exact, 2),
        "ms_per_sample": round(elapsed_ms, 1),
    }
    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"  Samples    : {len(refs)}")
    print(f"  BLEU       : {bleu:.2f}")
    print(f"  Token F1   : {f1:.2f}%")
    print(f"  Exact Match: {exact:.2f}%")
    print(f"  ms/sample  : {elapsed_ms:.1f}")
    return result


def run_marian(gpu_id: int, result_queue):
    from transformers import MarianMTModel, MarianTokenizer
    device = f"cuda:{gpu_id}"
    test = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=["english", "lunyoro"])
    results = []

    for direction, src_col, tgt_col in [("en2lun", "english", "lunyoro"), ("lun2en", "lunyoro", "english")]:
        print(f"[GPU {gpu_id}] Loading MarianMT {direction}...")
        tok = MarianTokenizer.from_pretrained(str(MODEL_DIR / direction))
        mdl = MarianMTModel.from_pretrained(str(MODEL_DIR / direction)).to(device).eval()
        srcs, refs = test[src_col].tolist(), test[tgt_col].tolist()
        hyps, t0 = [], time.time()
        for i in range(0, len(srcs), BATCH):
            batch = srcs[i:i+BATCH]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = mdl.generate(**inputs, num_beams=4, max_length=256)
            hyps += [tok.decode(o, skip_special_tokens=True) for o in out]
        elapsed = (time.time() - t0) * 1000 / len(srcs)
        results.append(compute_metrics(refs, hyps, f"MarianMT {direction}", elapsed))
        del mdl
        torch.cuda.empty_cache()

    result_queue.put(results)


def run_nllb(gpu_id: int, result_queue):
    from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
    device = f"cuda:{gpu_id}"
    test = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=["english", "lunyoro"])
    results = []

    for direction, src_col, tgt_col, src_lang, tgt_lang in [
        ("nllb_en2lun", "english",  "lunyoro", "eng_Latn", "run_Latn"),
        ("nllb_lun2en", "lunyoro",  "english", "run_Latn", "eng_Latn"),
    ]:
        print(f"[GPU {gpu_id}] Loading NLLB {direction}...")
        tok = NllbTokenizer.from_pretrained(str(MODEL_DIR / direction))
        mdl = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR / direction)).to(device).eval()
        tok.src_lang = src_lang
        tgt_id = tok.convert_tokens_to_ids(tgt_lang)
        srcs, refs = test[src_col].tolist(), test[tgt_col].tolist()
        hyps, t0 = [], time.time()
        for i in range(0, len(srcs), BATCH):
            batch = srcs[i:i+BATCH]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = mdl.generate(**inputs, forced_bos_token_id=tgt_id, num_beams=4, max_length=256)
            hyps += [tok.decode(o, skip_special_tokens=True) for o in out]
        elapsed = (time.time() - t0) * 1000 / len(srcs)
        results.append(compute_metrics(refs, hyps, f"NLLB {direction}", elapsed))
        del mdl
        torch.cuda.empty_cache()

    result_queue.put(results)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    test = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=["english", "lunyoro"])
    print(f"Test set: {len(test)} pairs")
    print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {BATCH}\n")

    q = mp.Queue()
    p0 = mp.Process(target=run_marian, args=(0, q))
    p1 = mp.Process(target=run_nllb,   args=(1, q))

    p0.start(); p1.start()
    p0.join();  p1.join()

    all_results = []
    while not q.empty():
        all_results.extend(q.get())

    # Sort by model name for consistent output
    all_results.sort(key=lambda x: x["model"])

    out = Path(__file__).parent / "eval_results_all.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'='*52}")
    print(f"Results saved → {out}")
    print(f"\n{'Model':<25} {'BLEU':>6} {'Token F1':>9} {'Exact%':>7} {'ms/s':>6}")
    print("-" * 57)
    for r in all_results:
        print(f"{r['model']:<25} {r['bleu']:>6.2f} {r['token_f1']:>8.2f}% {r['exact_match_pct']:>6.2f}% {r['ms_per_sample']:>6.1f}")
