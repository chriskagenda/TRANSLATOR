import pandas as pd, re, random

df = pd.read_csv("data/cleaned/english_nyoro_clean.csv").fillna("")
total = len(df)
print(f"Total pairs: {total:,}")

synthetic = df[~df["english"].str.startswith("[")].copy()
print(f"Synthetic/untagged pairs: {len(synthetic):,}")

issues = {
    "too_short_en":    (synthetic["english"].str.len() < 10).sum(),
    "too_short_lun":   (synthetic["lunyoro"].str.len() < 10).sum(),
    "repeated_words":  synthetic["english"].apply(lambda x: bool(re.search(r"\b(\w+)\s+\1\b", x, re.I))).sum(),
    "non_ascii":       synthetic["english"].apply(lambda x: sum(1 for c in x if ord(c)>127)/max(len(x),1) > 0.1).sum(),
    "very_long_en":    (synthetic["english"].str.split().str.len() > 50).sum(),
    "en_equals_lun":   (synthetic["english"].str.lower() == synthetic["lunyoro"].str.lower()).sum(),
}

print("\nQuality issues in synthetic pairs:")
for k, v in issues.items():
    pct = v / max(len(synthetic), 1) * 100
    status = "OK" if pct < 1 else ("WARN" if pct < 5 else "BAD")
    print(f"  [{status}] {k:<20} {v:>6,} ({pct:.1f}%)")

print("\nRandom sample of 5 synthetic pairs:")
sample = synthetic.sample(5, random_state=42)
for _, row in sample.iterrows():
    print(f"  EN:  {row['english'][:90]}")
    print(f"  LUN: {row['lunyoro'][:90]}")
    print()
