"""
Full improvement pipeline:
  1. Back-translate Lunyoro-only sentence_submissions → new pairs
  2. Quality-filter all synthetic pairs (clean_backtranslated logic)
  3. Rebuild training splits
  4. Fine-tune MarianMT from existing checkpoint (lower LR, more epochs, label_smoothing=0.2)
  5. Fine-tune NLLB from existing checkpoint

Run:
    python improve_and_retrain.py
"""
import os, re, time, sys
# Ensure all print() calls flush immediately — no buffering
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MarianDataset(Dataset):
    def __init__(self, df, tokenizer, src_col, tgt_col, max_len=256):
        self.tokenizer = tokenizer
        self.src = df[src_col].tolist()
        self.tgt = df[tgt_col].tolist()
        self.max_len = max_len

    def __len__(self): return len(self.src)

    def __getitem__(self, idx):
        src_enc = self.tokenizer(self.src[idx], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt")
        tgt_enc = self.tokenizer(text_target=self.tgt[idx], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt")
        labels = tgt_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": src_enc["input_ids"].squeeze(),
                "attention_mask": src_enc["attention_mask"].squeeze(),
                "labels": labels}


class NLLBDataset(Dataset):
    def __init__(self, df, tokenizer, src_col, tgt_col, src_lang, tgt_lang, max_len=128):
        self.tokenizer = tokenizer
        self.src = df[src_col].tolist(); self.tgt = df[tgt_col].tolist()
        self.src_lang = src_lang; self.tgt_lang = tgt_lang; self.max_len = max_len

    def __len__(self): return len(self.src)

    def __getitem__(self, idx):
        self.tokenizer.src_lang = self.src_lang
        src_enc = self.tokenizer(self.src[idx], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt")
        self.tokenizer.src_lang = self.tgt_lang
        tgt_enc = self.tokenizer(self.tgt[idx], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt")
        labels = tgt_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": src_enc["input_ids"].squeeze(),
                "attention_mask": src_enc["attention_mask"].squeeze(),
                "labels": labels}

BASE      = Path(__file__).parent
RAW       = BASE / "data" / "raw"
CLEAN_DIR = BASE / "data" / "cleaned"
OUT_DIR   = BASE / "data" / "training"
MODEL_DIR = BASE / "model"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Back-translate Lunyoro-only sentence_submissions
# ─────────────────────────────────────────────────────────────────────────────
def step1_back_translate():
    print("\n" + "="*60)
    print("STEP 1: Back-translating sentence_submissions")
    print("="*60)

    from transformers import MarianMTModel, MarianTokenizer

    # Collect all Lunyoro-only sentences from sentence_submissions
    lun_sentences = set()
    for fname in ["sentence_submissions_rows.csv", "sentence_submissions_rows (1).csv"]:
        p = RAW / fname
        if not p.exists():
            continue
        df = pd.read_csv(p).fillna("")
        for _, row in df.iterrows():
            s = str(row.get("translation_rutooro", "") or row.get("translation_runyoro", "")).strip()
            if len(s) >= 8:
                lun_sentences.add(s)

    print(f"  Unique Lunyoro sentences to back-translate: {len(lun_sentences)}")

    tok = MarianTokenizer.from_pretrained(str(MODEL_DIR / "lun2en"))
    mdl = MarianMTModel.from_pretrained(str(MODEL_DIR / "lun2en")).to(DEVICE).eval()

    sentences = list(lun_sentences)
    BATCH = 64
    synthetic = []

    for i in range(0, len(sentences), BATCH):
        batch = sentences[i:i+BATCH]
        inputs = tok(batch, return_tensors="pt", truncation=True,
                     padding=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(**inputs, num_beams=4, max_length=256,
                               no_repeat_ngram_size=3, repetition_penalty=1.3)
        for j, ids in enumerate(out):
            en  = tok.decode(ids, skip_special_tokens=True).strip()
            lun = batch[j].strip()
            if not en or not lun or en.lower() == lun.lower():
                continue
            if len(en) < 8 or len(lun) < 8:
                continue
            if len(en.split()) < len(lun.split()) * 0.25:
                continue
            words = en.lower().split()
            if len(words) > 3:
                bigrams = list(zip(words, words[1:]))
                if sum(1 for a,b in bigrams if a==b)/len(bigrams) > 0.3:
                    continue
            if sum(1 for c in en if ord(c)<128)/max(len(en),1) < 0.85:
                continue
            synthetic.append({"english": en, "lunyoro": lun})

        if (i // BATCH) % 5 == 0:
            print(f"  {min(i+BATCH, len(sentences))}/{len(sentences)} translated...")

    del mdl; torch.cuda.empty_cache()

    new_df = pd.DataFrame(synthetic)
    existing = pd.read_csv(CLEAN_DIR / "english_nyoro_clean.csv")
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["english","lunyoro"]).reset_index(drop=True)
    combined.to_csv(CLEAN_DIR / "english_nyoro_clean.csv", index=False)
    print(f"  Back-translated: {len(synthetic)} new pairs")
    print(f"  Total pairs: {len(existing):,} → {len(combined):,}")
    return len(synthetic)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Quality-filter synthetic pairs
# ─────────────────────────────────────────────────────────────────────────────
def step2_quality_filter():
    print("\n" + "="*60)
    print("STEP 2: Quality-filtering synthetic pairs")
    print("="*60)

    import unicodedata

    def is_bad(en: str, lun: str) -> bool:
        en, lun = str(en).strip(), str(lun).strip()
        if len(en) < 8 or len(lun) < 8: return True
        if len(lun.split()) < 2: return True
        if en.lower() == lun.lower(): return True
        en_w, lun_w = en.split(), lun.split()
        if len(en_w) < len(lun_w) * 0.25: return True
        if len(en_w) > 3:
            bigrams = list(zip(en_w, en_w[1:]))
            if sum(1 for a,b in bigrams if a.lower()==b.lower())/len(bigrams) > 0.3:
                return True
        if re.search(r'\b(\w+)\s+\1\s+\1\b', en, re.IGNORECASE): return True
        if sum(1 for c in en if ord(c)<128)/max(len(en),1) < 0.85: return True
        return False

    df = pd.read_csv(CLEAN_DIR / "english_nyoro_clean.csv").fillna("")
    before = len(df)
    mask = df.apply(lambda r: is_bad(r["english"], r["lunyoro"]), axis=1)
    df = df[~mask].reset_index(drop=True)
    df.to_csv(CLEAN_DIR / "english_nyoro_clean.csv", index=False)
    print(f"  Removed {mask.sum():,} bad pairs ({mask.sum()/before*100:.1f}%)")
    print(f"  Remaining: {len(df):,} pairs")
    return len(df)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Rebuild training splits
# ─────────────────────────────────────────────────────────────────────────────
def step3_rebuild_splits():
    print("\n" + "="*60)
    print("STEP 3: Rebuilding training splits")
    print("="*60)

    from prepare_training_data import build_corpus
    corpus = build_corpus()
    train, temp = train_test_split(corpus, test_size=0.1, random_state=42)
    val, test   = train_test_split(temp,   test_size=0.5, random_state=42)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(  OUT_DIR / "val.csv",   index=False)
    test.to_csv( OUT_DIR / "test.csv",  index=False)
    total = len(train)+len(val)+len(test)
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}  Total: {total:,}")
    return len(train)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Fine-tune MarianMT from existing checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def step4_finetune_marian():
    from transformers import MarianMTModel, MarianTokenizer, get_cosine_schedule_with_warmup
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    EPOCHS = 15
    BATCH  = 32
    LR     = 2e-5
    LABEL_SMOOTHING = 0.2
    NUM_GPUS = torch.cuda.device_count()

    train_df = pd.read_csv(OUT_DIR / "train.csv").fillna("")
    val_df   = pd.read_csv(OUT_DIR / "val.csv").fillna("")

    for direction, src_col, tgt_col in [("en2lun","english","lunyoro"), ("lun2en","lunyoro","english")]:
        print("\n" + "="*60)
        print(f"STEP 4: MarianMT {direction} — continuing from checkpoint")
        print(f"  Epochs={EPOCHS}  LR={LR}  LabelSmoothing={LABEL_SMOOTHING}  Batch={BATCH*max(NUM_GPUS,1)}")
        print("="*60)

        ckpt = str(MODEL_DIR / direction)
        tok  = MarianTokenizer.from_pretrained(ckpt)
        mdl  = MarianMTModel.from_pretrained(ckpt)
        mdl.config.label_smoothing_factor = LABEL_SMOOTHING
        mdl = mdl.to(DEVICE)
        if NUM_GPUS > 1:
            mdl = torch.nn.DataParallel(mdl)

        eff_batch = BATCH * max(NUM_GPUS, 1)
        train_ds = MarianDataset(train_df, tok, src_col, tgt_col)
        val_ds   = MarianDataset(val_df,   tok, src_col, tgt_col)
        train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=eff_batch, shuffle=False, num_workers=0, pin_memory=True)

        raw_mdl   = mdl.module if isinstance(mdl, torch.nn.DataParallel) else mdl
        optimizer = AdamW(raw_mdl.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

        best_val = float("inf")
        for epoch in range(1, EPOCHS+1):
            mdl.train()
            t_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                loss = mdl(**batch).loss
                if loss.dim() > 0: loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                t_loss += loss.item()
            t_loss /= len(train_loader)

            mdl.eval(); v_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    v_loss += mdl(**batch).loss.mean().item()
            v_loss /= len(val_loader)
            print(f"  Epoch {epoch}/{EPOCHS}  train={t_loss:.4f}  val={v_loss:.4f}")

            if v_loss < best_val:
                best_val = v_loss
                raw_mdl = mdl.module if isinstance(mdl, torch.nn.DataParallel) else mdl
                raw_mdl.save_pretrained(ckpt)
                tok.save_pretrained(ckpt)
                print(f"  ✓ Saved (val={v_loss:.4f})")

        del mdl; torch.cuda.empty_cache()
        print(f"  Done. Best val_loss={best_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Fine-tune NLLB from existing checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def step5_finetune_nllb():
    """
    Uses DistributedDataParallel (DDP) via torchrun for proper multi-GPU training.
    DDP works correctly with gradient checkpointing unlike DataParallel.
    Falls back to single GPU if only one available.
    """
    import subprocess, sys

    EPOCHS   = 8
    PER_GPU  = 8
    ACCUM    = 4
    LR       = 2e-5
    LABEL_SMOOTHING = 0.2
    NUM_GPUS = torch.cuda.device_count()

    # Write a standalone DDP training script that torchrun can launch
    ddp_script = BASE / "_nllb_ddp_worker.py"
    ddp_script.write_text(f'''
import os, sys
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup
import pandas as pd
from pathlib import Path

EPOCHS = {EPOCHS}; PER_GPU = {PER_GPU}; ACCUM = {ACCUM}; LR = {LR}; LS = {LABEL_SMOOTHING}
DATA_DIR  = Path(r"{OUT_DIR}")
MODEL_DIR = Path(r"{MODEL_DIR}")
LANG_EN   = "eng_Latn"

dist.init_process_group("gloo")   # gloo works on Windows; nccl requires Linux
rank       = dist.get_rank()
world_size = dist.get_world_size()
device     = torch.device(f"cuda:{{rank}}")
torch.cuda.set_device(device)

class NLLBDataset(Dataset):
    def __init__(self, df, tok, sc, tc, sl, tl, max_len=128):
        self.tok=tok; self.src=df[sc].tolist(); self.tgt=df[tc].tolist()
        self.sl=sl; self.tl=tl; self.max_len=max_len
    def __len__(self): return len(self.src)
    def __getitem__(self, idx):
        self.tok.src_lang=self.sl
        s=self.tok(self.src[idx],max_length=self.max_len,truncation=True,padding="max_length",return_tensors="pt")
        self.tok.src_lang=self.tl
        t=self.tok(self.tgt[idx],max_length=self.max_len,truncation=True,padding="max_length",return_tensors="pt")
        lbl=t["input_ids"].squeeze(); lbl[lbl==self.tok.pad_token_id]=-100
        return {{"input_ids":s["input_ids"].squeeze(),"attention_mask":s["attention_mask"].squeeze(),"labels":lbl}}

train_df = pd.read_csv(DATA_DIR/"train.csv").fillna("")
val_df   = pd.read_csv(DATA_DIR/"val.csv").fillna("")

for direction, sc, tc, sl, tl in [
    ("nllb_en2lun","english","lunyoro",LANG_EN,LANG_EN),
    ("nllb_lun2en","lunyoro","english",LANG_EN,LANG_EN),
]:
    ckpt = str(MODEL_DIR/direction)
    tok  = NllbTokenizer.from_pretrained(ckpt)
    mdl  = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    mdl.config.label_smoothing_factor = LS
    mdl.gradient_checkpointing_enable()
    mdl = mdl.to(device)
    mdl = DDP(mdl, device_ids=[rank], find_unused_parameters=False)

    train_ds = NLLBDataset(train_df,tok,sc,tc,sl,tl)
    val_ds   = NLLBDataset(val_df,tok,sc,tc,sl,tl)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader  = DataLoader(train_ds, batch_size=PER_GPU, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader    = DataLoader(val_ds,   batch_size=PER_GPU, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = AdamW(mdl.module.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader)//ACCUM)*EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps//10, total_steps)

    best_val = float("inf")
    for epoch in range(1, EPOCHS+1):
        train_sampler.set_epoch(epoch)
        mdl.train(); t_loss=0.0; optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch={{k:v.to(device) for k,v in batch.items()}}
            loss=mdl(**batch).loss
            if loss.dim()>0: loss=loss.mean()
            (loss/ACCUM).backward(); t_loss+=loss.item()
            if (step+1)%ACCUM==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
        optimizer.step(); optimizer.zero_grad()
        t_loss/=len(train_loader)
        if rank==0:
            mdl.eval(); v_loss=0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch={{k:v.to(device) for k,v in batch.items()}}
                    v_loss+=mdl(**batch).loss.mean().item()
            v_loss/=len(val_loader)
            print(f"  [{{direction}}] Epoch {{epoch}}/{{EPOCHS}}  train={{t_loss:.4f}}  val={{v_loss:.4f}}", flush=True)
            if v_loss<best_val:
                best_val=v_loss
                mdl.module.save_pretrained(ckpt); tok.save_pretrained(ckpt)
                print(f"  ✓ Saved (val={{v_loss:.4f}})", flush=True)
    if rank==0:
        print(f"  Done. Best val_loss={{best_val:.4f}}", flush=True)
    torch.cuda.empty_cache()

dist.destroy_process_group()
''')

    if NUM_GPUS >= 2:
        print(f"\n{'='*60}")
        print(f"STEP 5: NLLB — DDP across {NUM_GPUS} GPUs")
        print(f"  Epochs={EPOCHS}  LR={LR}  LabelSmoothing={LABEL_SMOOTHING}  PER_GPU={PER_GPU}")
        print("="*60)
        result = subprocess.run(
            [sys.executable, "-m", "torch.distributed.run",
             "--nproc_per_node", str(NUM_GPUS),
             "--master_port", "29500",
             str(ddp_script)],
            cwd=str(BASE)
        )
        if result.returncode != 0:
            print("  DDP failed, falling back to single GPU...")
            _nllb_single_gpu()
    else:
        _nllb_single_gpu()

    ddp_script.unlink(missing_ok=True)


def _nllb_single_gpu():
    from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    EPOCHS=8; PER_GPU=8; ACCUM=4; LR=2e-5; LABEL_SMOOTHING=0.2
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    device = f"cuda:{free.index(max(free))}"
    print(f"  Single GPU fallback on {device}")

    train_df = pd.read_csv(OUT_DIR/"train.csv").fillna("")
    val_df   = pd.read_csv(OUT_DIR/"val.csv").fillna("")

    for direction, sc, tc, sl, tl in [
        ("nllb_en2lun","english","lunyoro","eng_Latn","eng_Latn"),
        ("nllb_lun2en","lunyoro","english","eng_Latn","eng_Latn"),
    ]:
        print(f"\nNLLB {direction} [{device}]...")
        ckpt = str(MODEL_DIR/direction)
        tok  = NllbTokenizer.from_pretrained(ckpt)
        mdl  = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
        mdl.config.label_smoothing_factor = LABEL_SMOOTHING
        mdl.gradient_checkpointing_enable()
        mdl = mdl.to(device).train()

        ds_train = NLLBDataset(train_df, tok, sc, tc, sl, tl)
        ds_val   = NLLBDataset(val_df,   tok, sc, tc, sl, tl)
        tl_loader = DataLoader(ds_train, batch_size=PER_GPU, shuffle=True,  num_workers=0, pin_memory=True)
        vl_loader = DataLoader(ds_val,   batch_size=PER_GPU, shuffle=False, num_workers=0, pin_memory=True)

        opt = AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
        total = (len(tl_loader)//ACCUM)*EPOCHS
        sch = get_cosine_schedule_with_warmup(opt, total//10, total)
        best_val = float("inf")

        for epoch in range(1, EPOCHS+1):
            mdl.train(); t_loss=0.0; opt.zero_grad()
            for step, batch in enumerate(tl_loader):
                batch={k:v.to(device) for k,v in batch.items()}
                loss=mdl(**batch).loss
                if loss.dim()>0: loss=loss.mean()
                (loss/ACCUM).backward(); t_loss+=loss.item()
                if (step+1)%ACCUM==0:
                    torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
                    opt.step(); sch.step(); opt.zero_grad()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
            opt.step(); opt.zero_grad(); t_loss/=len(tl_loader)
            mdl.eval(); v_loss=0.0
            with torch.no_grad():
                for batch in vl_loader:
                    batch={k:v.to(device) for k,v in batch.items()}
                    v_loss+=mdl(**batch).loss.mean().item()
            v_loss/=len(vl_loader)
            print(f"  Epoch {epoch}/{EPOCHS}  train={t_loss:.4f}  val={v_loss:.4f}", flush=True)
            if v_loss<best_val:
                best_val=v_loss; mdl.save_pretrained(ckpt); tok.save_pretrained(ckpt)
                print(f"  ✓ Saved (val={v_loss:.4f})", flush=True)
        del mdl; torch.cuda.empty_cache()
        print(f"  Done. Best val_loss={best_val:.4f}", flush=True)

    train_df = pd.read_csv(OUT_DIR / "train.csv").fillna("")
    val_df   = pd.read_csv(OUT_DIR / "val.csv").fillna("")

    for direction, src_col, tgt_col, sl, tl in [
        ("nllb_en2lun","english","lunyoro",LANG_EN,LANG_LUN),
        ("nllb_lun2en","lunyoro","english",LANG_LUN,LANG_EN),
    ]:
        print("\n" + "="*60)
        print(f"STEP 5: NLLB {direction} — continuing from checkpoint")
        eff = PER_GPU * max(NUM_GPUS,1) * ACCUM
        print(f"  Epochs={EPOCHS}  LR={LR}  LabelSmoothing={LABEL_SMOOTHING}  EffBatch={eff}")
        print("="*60)

        ckpt = str(MODEL_DIR / direction)
        tok  = NllbTokenizer.from_pretrained(ckpt)
        mdl  = AutoModelForSeq2SeqLM.from_pretrained(ckpt)   # ← load existing weights
        mdl.config.label_smoothing_factor = LABEL_SMOOTHING
        mdl = mdl.to(DEVICE)
        mdl.gradient_checkpointing_enable()
        if NUM_GPUS > 1:
            mdl = torch.nn.DataParallel(mdl)

        train_ds = NLLBDataset(train_df, tok, src_col, tgt_col, sl, tl)
        val_ds   = NLLBDataset(val_df,   tok, src_col, tgt_col, sl, tl)
        train_loader = DataLoader(train_ds, batch_size=PER_GPU, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=PER_GPU, shuffle=False, num_workers=0, pin_memory=True)

        raw_mdl   = mdl.module if isinstance(mdl, torch.nn.DataParallel) else mdl
        optimizer = AdamW(raw_mdl.parameters(), lr=LR, weight_decay=0.01)
        total_steps = (len(train_loader) // ACCUM) * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

        best_val = float("inf")
        for epoch in range(1, EPOCHS+1):
            mdl.train(); t_loss = 0.0; optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                loss = mdl(**batch).loss
                if loss.dim() > 0: loss = loss.mean()
                (loss / ACCUM).backward()
                t_loss += loss.item()
                if (step+1) % ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
            # flush remaining
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            t_loss /= len(train_loader)

            mdl.eval(); v_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    v_loss += mdl(**batch).loss.mean().item()
            v_loss /= len(val_loader)
            print(f"  Epoch {epoch}/{EPOCHS}  train={t_loss:.4f}  val={v_loss:.4f}", flush=True)

            if v_loss < best_val:
                best_val = v_loss
                raw_mdl = mdl.module if isinstance(mdl, torch.nn.DataParallel) else mdl
                raw_mdl.save_pretrained(ckpt)
                tok.save_pretrained(ckpt)
                print(f"  ✓ Saved (val={v_loss:.4f})", flush=True)

        del mdl; torch.cuda.empty_cache()
        print(f"  Done. Best val_loss={best_val:.4f}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Push all models to HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
def step6_push_to_hf():
    print("\n" + "="*60)
    print("STEP 6: Pushing models to HuggingFace")
    print("="*60)

    from huggingface_hub import HfApi
    api = HfApi()

    HF_REPOS = {
        "en2lun":      "keithtwesigye/lunyoro-en2lun",
        "lun2en":      "keithtwesigye/lunyoro-lun2en",
        "nllb_en2lun": "keithtwesigye/lunyoro-nllb_en2lun",
        "nllb_lun2en": "keithtwesigye/lunyoro-nllb_lun2en",
    }

    for local_name, repo_id in HF_REPOS.items():
        local_path = MODEL_DIR / local_name
        if not local_path.exists():
            print(f"  ✗ {local_name} not found, skipping")
            continue
        print(f"  ↑ Pushing {local_name} → {repo_id} ...")
        try:
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Retrain on 71,870 pairs: checkpoint fine-tune, label_smoothing=0.2, back-translation augmentation",
                ignore_patterns=["*.cache", "__pycache__", ".cache"],
            )
            print(f"  ✓ {repo_id} pushed")
        except Exception as e:
            print(f"  ✗ Failed to push {repo_id}: {e}")

    # Also push updated dataset
    print(f"\n  ↑ Pushing dataset → keithtwesigye/lunyoro-rutooro-parallel ...")
    try:
        api.upload_folder(
            folder_path=str(OUT_DIR),
            repo_id="keithtwesigye/lunyoro-rutooro-parallel",
            repo_type="dataset",
            commit_message="Updated training splits: 71,870 pairs with back-translation augmentation",
            ignore_patterns=["*.cache", "__pycache__"],
        )
        print("  ✓ Dataset pushed")
    except Exception as e:
        print(f"  ✗ Failed to push dataset: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Push everything to Git LFS
# ─────────────────────────────────────────────────────────────────────────────
def step7_push_to_git():
    import subprocess

    print("\n" + "="*60)
    print("STEP 7: Pushing to Git LFS repo")
    print("="*60)

    repo_root = str(BASE.parent)  # lunyoro-translator/

    def run(cmd, cwd=repo_root):
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"  {result.stdout.strip()}")
        if result.returncode != 0 and result.stderr.strip():
            print(f"  WARN: {result.stderr.strip()[:300]}")
        return result.returncode

    # Stage all model weights, data, and scripts
    print("  Staging changes...")
    run(["git", "add",
         "backend/model/en2lun/",
         "backend/model/lun2en/",
         "backend/model/nllb_en2lun/",
         "backend/model/nllb_lun2en/",
         "backend/data/cleaned/",
         "backend/data/training/",
         "backend/improve_and_retrain.py",
         "backend/clean_unprocessed_raw.py",
         "backend/run_eval.py",
         "backend/eval_all_parallel.py",
    ])

    # Check if there's anything to commit
    status = subprocess.run(["git", "status", "--porcelain"],
                            cwd=repo_root, capture_output=True, text=True)
    if not status.stdout.strip():
        print("  Nothing to commit — repo already up to date")
        return

    print("  Committing...")
    run(["git", "commit", "-m",
         "Retrain: checkpoint fine-tune on 71,870 pairs, back-translation augmentation, label_smoothing=0.2"])

    print("  Pushing (LFS + regular)...")
    rc = run(["git", "push", "origin", "main"])
    if rc == 0:
        print("  ✓ Git push complete")
    else:
        print("  ✗ Git push failed — check credentials or run manually: git push origin main")


if __name__ == "__main__":
    # Force unbuffered output so epoch logs appear immediately
    sys.stdout.reconfigure(line_buffering=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip steps 1-3 (data already prepared)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only push to HF")
    args = parser.parse_args()

    t0 = time.time()
    print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    if not args.skip_data and not args.skip_train:
        new_bt    = step1_back_translate()
        remaining = step2_quality_filter()
        train_n   = step3_rebuild_splits()
    else:
        print("Skipping data steps")
        train_n = len(pd.read_csv(OUT_DIR / "train.csv"))
        print(f"  Training pairs: {train_n:,}")

    if not args.skip_train:
        step4_finetune_marian()
        step5_finetune_nllb()

    step6_push_to_hf()
    step7_push_to_git()

    elapsed = (time.time()-t0)/3600
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed:.1f}h")
    print(f"  Final training pairs : {train_n:,}")
    print(f"  Models pushed to     : huggingface.co/keithtwesigye")
    print(f"  Code/data pushed to  : github.com/chriskagenda/TRANSLATOR")
    print(f"  Run python run_eval.py to benchmark")
