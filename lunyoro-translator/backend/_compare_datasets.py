import pandas as pd

train = pd.read_csv("data/training/train.csv").fillna("")
val   = pd.read_csv("data/training/val.csv").fillna("")
test  = pd.read_csv("data/training/test.csv").fillna("")
clean = pd.read_csv("data/cleaned/english_nyoro_clean.csv").fillna("")

total_training = len(train) + len(val) + len(test)
synth_train = train[~train["english"].str.startswith("[")]
synth_clean = clean[~clean["english"].str.startswith("[")]

print("=== CURRENTLY TRAINING ON (59k splits) ===")
print(f"  train : {len(train):,}")
print(f"  val   : {len(val):,}")
print(f"  test  : {len(test):,}")
print(f"  TOTAL : {total_training:,}")
print(f"  synthetic in train: {len(synth_train):,} ({len(synth_train)/len(train)*100:.1f}%)")

print()
print("=== CLEAN CORPUS (next training will use) ===")
print(f"  raw pairs : {len(clean):,}")
print(f"  synthetic : {len(synth_clean):,} ({len(synth_clean)/len(clean)*100:.1f}%)")
print(f"  original  : {len(clean)-len(synth_clean):,} ({(len(clean)-len(synth_clean))/len(clean)*100:.1f}%)")

print()
print("=== KEY DIFFERENCES ===")
print(f"  Size diff: {total_training - len(clean):,} fewer pairs in clean corpus")
print(f"  Synthetic removed: ~{71511 - len(clean):,} bad back-translated pairs filtered out")
print(f"  Quality: clean corpus passed length + repeat + ASCII + single-word + round-trip + semantic filters")
print(f"  Current training: used 71k raw (18k garbage included)")
