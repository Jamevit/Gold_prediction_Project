"""
CS322 - Gold Trend Prediction with an LSTM neural network
=========================================================
Predicts the next trading day's gold close (USD/oz) from the previous 60 days.

Why returns instead of raw prices?
    Gold went from ~$1,100 (2016) to ~$5,300 (Jan 2026). A network trained on
    2016-2023 prices has never seen $4,000+, so a min-max-scaled price model
    saturates on the 2025 test set. Instead each input window is expressed as
    log-returns, and the network predicts tomorrow's log-return r:
        predicted_close(t+1) = close(t) * exp(r_hat)
    This keeps every input in the same range no matter the price level.

Data split (chronological, no shuffling across time):
    train : 2016-02 .. 2023-12
    valid : 2024-01 .. 2024-12   (early stopping)
    test  : 2025-01 .. 2026-01-30 (never seen during training)

Baselines it must beat to be useful:
    * Naive / random walk : close(t+1) = close(t)
    * Least-squares lag-1 : close(t+1) = a + b * close(t)   (fit on train)

Run:  python gold_lstm.py            (needs: numpy pandas matplotlib torch)
"""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "gold_historical_data.csv")
OUT = os.path.join(HERE, "lstm_outputs")
os.makedirs(OUT, exist_ok=True)

WINDOW = 60          # days of history fed to the LSTM
HIDDEN = 32
LAYERS = 1
DROPOUT = 0.2
LR = 1e-3
BATCH = 64
MAX_EPOCHS = 300
PATIENCE = 30
SEED = 42
TRAIN_END = "2023-12-31"
VALID_END = "2024-12-31"
FORECAST_TO = "2026-11-10"      # recursive forecast horizon (today)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


set_seed(SEED)
torch.set_num_threads(2)

# ---------------------------------------------------------------- data
df = pd.read_csv(CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
close = df["Close"].to_numpy(dtype=float)
dates = df["Date"]
logret = np.diff(np.log(close))               # r[i] = ln(C[i+1]/C[i]), length n-1

# Standardise returns with TRAIN statistics only (no look-ahead)
n_train_ret = int((dates <= TRAIN_END).sum()) - 1
mu, sd = logret[:n_train_ret].mean(), logret[:n_train_ret].std()
z = (logret - mu) / sd

# Sample k: input = z[k-WINDOW:k] (returns up to day k), target = z[k] (return day k -> k+1)
# The close being predicted is close[k+1]; the last known close is close[k].
X, Y, tgt_idx = [], [], []
for k in range(WINDOW, len(z)):
    X.append(z[k - WINDOW:k]); Y.append(z[k]); tgt_idx.append(k + 1)
X = np.array(X, dtype=np.float32)[..., None]
Y = np.array(Y, dtype=np.float32)
tgt_idx = np.array(tgt_idx)
tgt_dates = dates.iloc[tgt_idx].to_numpy()

tr = tgt_dates <= np.datetime64(TRAIN_END)
va = (tgt_dates > np.datetime64(TRAIN_END)) & (tgt_dates <= np.datetime64(VALID_END))
te = tgt_dates > np.datetime64(VALID_END)
print(f"samples  train={tr.sum()}  valid={va.sum()}  test={te.sum()}")


# ---------------------------------------------------------------- model
class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, num_layers=LAYERS, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


model = LSTMRegressor()
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()          # least-squares loss, same criterion as the regression part

Xt, Yt = torch.tensor(X), torch.tensor(Y)
best, best_state, wait, history = np.inf, None, 0, []
idx_tr = np.where(tr)[0]
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    np.random.shuffle(idx_tr)
    for b in range(0, len(idx_tr), BATCH):
        bi = idx_tr[b:b + BATCH]
        opt.zero_grad()
        loss = loss_fn(model(Xt[bi]), Yt[bi])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        tl = loss_fn(model(Xt[tr]), Yt[tr]).item()
        vl = loss_fn(model(Xt[va]), Yt[va]).item()
    history.append((tl, vl))
    if vl < best - 1e-6:
        best, wait = vl, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            break
model.load_state_dict(best_state)
model.eval()
print(f"stopped at epoch {epoch}, best valid MSE (standardised) = {best:.4f}")

with torch.no_grad():
    z_hat = model(Xt).numpy()
r_hat = z_hat * sd + mu
prev_close = close[tgt_idx - 1]
pred_lstm = prev_close * np.exp(r_hat)
actual = close[tgt_idx]
pred_naive = prev_close

# Least-squares lag-1 baseline: solve normal equations (A^T A) beta = A^T y on train
A = np.column_stack([np.ones(tr.sum()), prev_close[tr]])
beta = np.linalg.solve(A.T @ A, A.T @ actual[tr])
pred_ls = beta[0] + beta[1] * prev_close


# ---------------------------------------------------------------- metrics
def metrics(y, p, prev):
    e = y - p
    sse = float(np.sum(e ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    dir_acc = float(np.mean(np.sign(p - prev) == np.sign(y - prev)) * 100)
    return dict(RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)),
                MAPE=np.mean(np.abs(e / y)) * 100, R2=1 - sse / sst, SSE=sse,
                DirAcc=dir_acc)


rows = []
for split, m in [("valid 2024", va), ("test 2025-Jan2026", te)]:
    for name, p in [("LSTM", pred_lstm), ("Naive (C_t+1 = C_t)", pred_naive),
                    ("LS linear lag-1", pred_ls)]:
        rows.append({"split": split, "model": name,
                     **metrics(actual[m], p[m], prev_close[m])})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "lstm_metrics.csv"), index=False)
pd.set_option("display.width", 140)
print(res.round(4).to_string(index=False))
# note: naive has no direction prediction (always 0 change) -> DirAcc not meaningful for it

pd.DataFrame({"Date": tgt_dates[te], "Actual": actual[te], "LSTM": pred_lstm[te],
              "Naive": pred_naive[te], "LS_lag1": pred_ls[te]}).to_csv(
    os.path.join(OUT, "lstm_test_predictions.csv"), index=False)

# ---------------------------------------------------------------- next-day + recursive forecast
last_z = torch.tensor(z[-WINDOW:], dtype=torch.float32)[None, :, None]
with torch.no_grad():
    r_next = model(last_z).item() * sd + mu
next_day = close[-1] * np.exp(r_next)
print(f"\nLast close {dates.iloc[-1].date()}: {close[-1]:.2f}")
print(f"LSTM next-day forecast (2026-02-02): {next_day:.2f}")

fdates = pd.bdate_range(dates.iloc[-1] + pd.Timedelta(days=1), FORECAST_TO)
window = list(z[-WINDOW:])
path, c = [], close[-1]
with torch.no_grad():
    for _ in fdates:
        zz = model(torch.tensor(window[-WINDOW:], dtype=torch.float32)[None, :, None]).item()
        c = c * np.exp(zz * sd + mu)
        path.append(c)
        window.append(zz)
fc = pd.DataFrame({"Date": fdates, "LSTM_recursive_forecast": path})
fc.to_csv(os.path.join(OUT, "lstm_forecast_to_nov2026.csv"), index=False)
print(f"LSTM recursive forecast for {FORECAST_TO}: {path[-1]:.2f}  ({len(fdates)} business days ahead)")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})

fig, ax = plt.subplots(figsize=(8, 4))
h = np.array(history)
ax.plot(h[:, 0], label="train MSE"); ax.plot(h[:, 1], label="validation MSE")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardised returns)")
ax.set_title("CS322 - LSTM training curve (early stopping)"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_lstm_training.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(tgt_dates[te], actual[te], color="black", lw=1.6, label="Actual close")
ax.plot(tgt_dates[te], pred_lstm[te], color="tab:orange", lw=1.1, label="LSTM (next-day)")
ax.plot(tgt_dates[te], pred_ls[te], color="tab:blue", lw=0.9, ls="--", label="LS linear lag-1")
ax.set_ylabel("USD / oz"); ax.legend()
ax.set_title("CS322 - Next-day gold price prediction on unseen test data (2025 - Jan 2026)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_lstm_test.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(dates, close, color="black", lw=1, label="Historical close")
ax.plot(fc["Date"], fc["LSTM_recursive_forecast"], color="tab:orange", lw=1.5,
        label="LSTM recursive forecast")
ax.axvline(dates.iloc[-1], color="grey", ls=":")
# Real observed price for checking the forecast (Fortune, 9 Nov 2026: ~$4,123/oz)
ax.scatter([pd.Timestamp("2026-11-09")], [4123], color="tab:red", zorder=5,
           label="Actual price 9 Nov 2026 (~$4,123)")
ax.set_ylabel("USD / oz"); ax.legend(loc="upper left")
ax.set_title("CS322 - LSTM multi-step forecast Feb - Nov 2026")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_lstm_forecast.png")); plt.close(fig)

torch.save(model.state_dict(), os.path.join(OUT, "gold_lstm.pt"))
print(f"\nSaved results to {OUT}")
