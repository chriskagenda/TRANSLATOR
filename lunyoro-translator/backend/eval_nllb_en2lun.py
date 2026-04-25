"""Evaluate only nllb_en2lun on the GPU with most free memory."""
import json, time, os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, pandas as pd
from pathlib import Path
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import BLEU

DATA_DIR  = Path(__file__).parent / "data" / "training"
MODEL_DIR = Path(__file__).parent / "model"
BATCH     = 32

def bleu_score(refs, hyps): return BLEU(effective_order=True).corpus_score(hyps, [refs]).score
def token_f1(ref, hyp):
    r, h = set(ref.lower().split()), set(hyp.lower().split())
    if not r or not h: return 0.0
    p = len(r&h)/len(h); rec = len(r&h)/len(r)
    return 2*p*rec/(p+rec) if (p+rec) else 0.0

# Pick GPU with most free memory (avoids the one training nllb_lun2en)
free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
device = f"cuda:{free.index(max(free))}"
print(f"Using {device} ({max(free)//1024**3}GB free)", flush=True)

test = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=["english","lunyoro"])
print(f"Test pairs: {len(test)}\n", flush=True)

direction, sc, tc, sl, tl = "nllb_en2lun", "english", "lunyoro", "eng_Latn", "run_Latn"
print(f"Evaluating NLLB {direction} ...", flush=True)
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
    if (i // BATCH) % 10 == 0:
        print(f"  {min(i+BATCH, len(srcs))}/{len(srcs)} ...", flush=True)

ms  = (time.time()-t0)*1000/len(srcs)
b   = bleu_score(refs, hyps)
f1  = sum(token_f1(r,h) for r,h in zip(refs,hyps))/len(refs)*100
ex  = sum(r.strip().lower()==h.strip().lower() for r,h in zip(refs,hyps))/len(refs)*100

print(f"\nNLLB {direction}:  BLEU={b:.2f}  F1={f1:.2f}%  Exact={ex:.2f}%  {ms:.1f}ms/s", flush=True)

out_f = Path(__file__).parent / "eval_nllb_en2lun_results.json"
out_f.write_text(json.dumps({"model": f"NLLB {direction}", "bleu": round(b,2),
    "token_f1": round(f1,2), "exact_match_pct": round(ex,2), "ms_per_sample": round(ms,1)}, indent=2))
print(f"Saved → {out_f.name}", flush=True)
