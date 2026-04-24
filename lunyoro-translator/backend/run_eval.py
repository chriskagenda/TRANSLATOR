"""Run eval on all 4 models. Marian on cuda:0, NLLB on cuda:1."""
import json, time, os
from pathlib import Path
import pandas as pd
import torch
from sacrebleu.metrics import BLEU

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DATA_DIR  = Path(__file__).parent / "data" / "training"
MODEL_DIR = Path(__file__).parent / "model"
BATCH     = 64


def bleu_score(refs, hyps):
    return BLEU(effective_order=True).corpus_score(hyps, [refs]).score


def token_f1(ref, hyp):
    r, h = set(ref.lower().split()), set(hyp.lower().split())
    if not r or not h:
        return 0.0
    p = len(r & h) / len(h)
    rec = len(r & h) / len(r)
    return 2 * p * rec / (p + rec) if (p + rec) else 0.0


def report(name, refs, hyps, elapsed_ms):
    b  = bleu_score(refs, hyps)
    f  = sum(token_f1(r, h) for r, h in zip(refs, hyps)) / len(refs) * 100
    ex = sum(r.strip().lower() == h.strip().lower() for r, h in zip(refs, hyps)) / len(refs) * 100
    print(f"  BLEU={b:.2f}  F1={f:.2f}%  Exact={ex:.2f}%  {elapsed_ms:.1f}ms/s")
    return {"model": name, "samples": len(refs), "bleu": round(b, 2),
            "token_f1": round(f, 2), "exact_match_pct": round(ex, 2),
            "ms_per_sample": round(elapsed_ms, 1)}


def main():
    test = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=["english", "lunyoro"])
    print(f"Test: {len(test)} pairs | GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}\n")
    results = []

    # Detect free GPU (Ollama may occupy GPU 0)
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    gpu = free.index(max(free))
    print(f"Using cuda:{gpu} (most free: {max(free)//1024**3}GB available)\n")
    device = f"cuda:{gpu}"

    # ── MarianMT ─────────────────────────────────────────────────────────────
    from transformers import MarianMTModel, MarianTokenizer
    for direction, sc, tc in [("en2lun", "english", "lunyoro"), ("lun2en", "lunyoro", "english")]:
        print(f"MarianMT {direction} [{device}] ...")
        tok = MarianTokenizer.from_pretrained(str(MODEL_DIR / direction))
        mdl = MarianMTModel.from_pretrained(str(MODEL_DIR / direction)).to(device).eval()
        srcs, refs = test[sc].tolist(), test[tc].tolist()
        hyps, t0 = [], time.time()
        for i in range(0, len(srcs), BATCH):
            inp = tok(srcs[i:i+BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = mdl.generate(**inp, num_beams=4, max_length=256)
            hyps += [tok.decode(o, skip_special_tokens=True) for o in out]
        results.append(report(f"MarianMT {direction}", refs, hyps, (time.time()-t0)*1000/len(srcs)))
        del mdl
        torch.cuda.empty_cache()

    # ── NLLB ─────────────────────────────────────────────────────────────────
    from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
    for direction, sc, tc, sl, tl in [
        ("nllb_en2lun", "english", "lunyoro", "eng_Latn", "run_Latn"),
        ("nllb_lun2en", "lunyoro", "english", "run_Latn", "eng_Latn"),
    ]:
        print(f"NLLB {direction} [{device}] ...")
        tok = NllbTokenizer.from_pretrained(str(MODEL_DIR / direction))
        mdl = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR / direction)).to(device).eval()
        tok.src_lang = sl
        tgt_id = tok.convert_tokens_to_ids(tl)
        srcs, refs = test[sc].tolist(), test[tc].tolist()
        hyps, t0 = [], time.time()
        for i in range(0, len(srcs), BATCH):
            inp = tok(srcs[i:i+BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = mdl.generate(**inp, forced_bos_token_id=tgt_id, num_beams=4, max_length=256)
            hyps += [tok.decode(o, skip_special_tokens=True) for o in out]
        results.append(report(f"NLLB {direction}", refs, hyps, (time.time()-t0)*1000/len(srcs)))
        del mdl
        torch.cuda.empty_cache()

    Path(__file__).parent.joinpath("eval_results_all.json").write_text(json.dumps(results, indent=2))

    print("\n" + "="*57)
    print(f"{'Model':<25} {'BLEU':>6} {'Token F1':>9} {'Exact%':>7} {'ms/s':>6}")
    print("-"*57)
    for r in results:
        print(f"{r['model']:<25} {r['bleu']:>6.2f} {r['token_f1']:>8.2f}% {r['exact_match_pct']:>6.2f}% {r['ms_per_sample']:>6.1f}")
    print("="*57)
    print("Saved → eval_results_all.json")


if __name__ == "__main__":
    main()
