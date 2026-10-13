"""Generate a synthetic, reproducible credit-card fraud dataset."""
import json
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 20000

merchants = ["grocery", "restaurant", "electronics", "travel", "gas", "online", "jewelry"]
merchant_idx = RNG.integers(0, len(merchants), N)
hour = RNG.integers(0, 24, N)
day_of_week = RNG.integers(0, 7, N)
amount = np.round(np.exp(RNG.normal(3.1, 1.15, N)), 2)
distance_from_home = np.round(np.abs(RNG.normal(12, 22, N)), 1)
distance_from_last_tx = np.round(np.abs(RNG.normal(7, 14, N)), 1)
ratio_to_median = np.round(np.clip(RNG.normal(1.0, 0.55, N), 0.05, 8.0), 2)
is_chip = RNG.choice([0, 1], N, p=[0.12, 0.88])
is_pin_used = RNG.choice([0, 1], N, p=[0.28, 0.72])
velocity_24h = RNG.poisson(2.2, N)
foreign = RNG.choice([0, 1], N, p=[0.90, 0.10])
card_present = RNG.choice([0, 1], N, p=[0.18, 0.82])
account_age_months = RNG.integers(1, 120, N)

is_online = (merchant_idx == 5).astype(int)
is_jewelry = (merchant_idx == 6).astype(int)
night = ((hour < 6) | (hour >= 23)).astype(int)
far_home = (distance_from_home > 40).astype(int)
far_last = (distance_from_last_tx > 25).astype(int)
high_velocity = (velocity_24h >= 5).astype(int)
new_acct = (account_age_months < 6).astype(int)
high_amount = (amount > 250).astype(int)

score = (
    0.9 * high_amount
    + 1.4 * foreign
    + 1.1 * (1 - is_chip)
    + 0.7 * (1 - is_pin_used)
    + 1.0 * is_online
    + 1.1 * is_jewelry
    + 0.7 * night
    + 0.7 * far_home
    + 0.5 * far_last
    + 0.6 * high_velocity
    + 0.7 * new_acct
    + 0.3 * (1 - card_present)
)

p = 1 / (1 + np.exp(-(score - 4.6) * 1.0))
fraud = (RNG.random(N) < p).astype(int)

df = pd.DataFrame({
    "transaction_id": [f"TX{i:06d}" for i in range(N)],
    "amount": amount,
    "hour": hour,
    "day_of_week": day_of_week,
    "merchant_category": merchant_idx,
    "merchant_name": [merchants[i] for i in merchant_idx],
    "distance_from_home": distance_from_home,
    "distance_from_last_tx": distance_from_last_tx,
    "ratio_to_median": ratio_to_median,
    "is_chip": is_chip,
    "is_pin_used": is_pin_used,
    "velocity_24h": velocity_24h,
    "foreign_transaction": foreign,
    "card_present": card_present,
    "account_age_months": account_age_months,
    "fraud": fraud,
})

out = "data/transactions.csv"
import os
os.makedirs("data", exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows, fraud_rate={fraud.mean():.4f} ({fraud.sum()} fraud)")

schema = {
    "merchants": merchants,
    "feature_order": [
        "amount", "hour", "day_of_week", "merchant_category",
        "distance_from_home", "distance_from_last_tx", "ratio_to_median",
        "is_chip", "is_pin_used", "velocity_24h", "foreign_transaction",
        "card_present", "account_age_months",
    ],
}
with open("data/schema.json", "w") as f:
    json.dump(schema, f, indent=2)
print("wrote data/schema.json")