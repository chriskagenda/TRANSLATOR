import os, time
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
import pandas as pd, torch
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu.metrics import BLEU

DATA_DIR  = Path(__file__).parent / "data" / "training"
MODEL_DIR = Path(__file__).parent / "model"
BATCH     = 64

DEVICE = "cpu"
print(f"Using CPU (leaving both GPUs free for NLLB training)")

def bleu(refs, hyps): return BLEU(effective_order=True).corpus_score(hyps, [refs]).score
def f1(ref, hyp):
    r, h = set(ref.lower().split()), set(hyp.lower().split())
    if not r or not h: return 0.0
    p = len(r&h)/len(h); rec = len(r&h)/len(r)
    return 2*p*rec/(p+rec) if (p+rec) else 0.0

test = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=["english","lunyoro"])
print(f"Test: {len(test)} pairs\n")

results = []
for direction, sc, tc in [("en2lun","english","lunyoro"), ("lun2en","lunyoro","english")]:
    tok = MarianTokenizer.from_pretrained(str(MODEL_DIR / direction))
    mdl = MarianMTModel.from_pretrained(str(MODEL_DIR / direction)).to(DEVICE).eval()
    srcs, refs = test[sc].tolist(), test[tc].tolist()
    hyps, t0 = [], time.time()
    for i in range(0, len(srcs), BATCH):
        inp = tok(srcs[i:i+BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(**inp, num_beams=4, max_length=256)
        hyps += [tok.decode(o, skip_special_tokens=True) for o in out]
    ms = (time.time()-t0)*1000/len(srcs)
    b  = bleu(refs, hyps)
    fi = sum(f1(r,h) for r,h in zip(refs,hyps))/len(refs)*100
    ex = sum(r.strip().lower()==h.strip().lower() for r,h in zip(refs,hyps))/len(refs)*100
    print(f"MarianMT {direction}:  BLEU={b:.2f}  F1={fi:.2f}%  Exact={ex:.2f}%  {ms:.1f}ms/s")
    results.append({"model": f"MarianMT {direction}", "bleu": round(b,2),
                    "token_f1": round(fi,2), "exact_match_pct": round(ex,2)})
    del mdl; torch.cuda.empty_cache()

import json
Path(__file__).parent.joinpath("eval_marian_results.json").write_text(json.dumps(results, indent=2))
print("\nSaved → eval_marian_results.json")
