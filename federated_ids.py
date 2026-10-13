"""
Federated Learning for Privacy-Preserving IoT Intrusion Detection
================================================================
Author : Mohammad Nafees (Ph.D. Scholar, ECE, MNNIT Allahabad)

Problem
-------
Five IoT gateways (e.g. a hospital, a factory, a campus, a smart-home provider
and a city network) each see only their OWN traffic, and each sees mostly a few
attack types. They cannot share raw traffic (privacy, law, business secrets).
A model trained alone at one gateway misses attacks it has never seen.

Idea
----
Federated Averaging (FedAvg): every round, each gateway trains the shared neural
network on its own data for one epoch and sends ONLY the model weights to a
server; the server averages them and sends the new model back. Raw data never
leaves the gateway.

Data: NSL-KDD (KDDTrain+ / KDDTest+), downloaded automatically.
"""

import os
import time
import copy
import warnings
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42
N_CLIENTS = 5          # IoT gateways
ALPHA = 0.3            # Dirichlet non-IID strength (smaller = each gateway more one-sided)
ROUNDS = 30            # federated rounds
HIDDEN = (64, 32)      # neural network size (same for every gateway)
rng = np.random.default_rng(SEED)
os.makedirs("data", exist_ok=True); os.makedirs("results", exist_ok=True)

# ------------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------------
URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
for f in ["KDDTrain+.txt", "KDDTest+.txt"]:
    if not os.path.exists(f"data/{f}"):
        print("Downloading", f); urllib.request.urlretrieve(URL + f, f"data/{f}")
COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]
FAMILY = {"DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb", "processtable", "udpstorm"],
          "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
          "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "named",
                  "sendmail", "snmpgetattack", "snmpguess", "worm", "xlock", "xsnoop"],
          "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel", "ps", "sqlattack", "xterm"]}
fam_of = {a: f for f, lst in FAMILY.items() for a in lst}
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
train_all = pd.read_csv("data/KDDTrain+.txt", names=COLS)
test_new = pd.read_csv("data/KDDTest+.txt", names=COLS)
for d in (train_all, test_new):
    d["y"] = d["label"].map(lambda a: 0 if a == "normal" else CLASSES.index(fam_of[a]))
# Held-out 20 % of KDDTrain+ = main test (same attack types); KDDTest+ = extra test (new attack types)
train, test_known = train_test_split(train_all, test_size=0.2, random_state=SEED, stratify=train_all["y"])

CAT = ["protocol_type", "service", "flag"]
NUM = [c for c in COLS[:-2] if c not in CAT]
VOCAB = {c: sorted(pd.concat([train_all[c], test_new[c]]).unique()) for c in CAT}   # public protocol names

# ------------------------------------------------------------------
# 2. Split the training data across gateways (non-IID, Dirichlet)
# ------------------------------------------------------------------
def dirichlet_split(y, n_clients, alpha):
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        share = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(share) * len(idx)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx, cuts)):
            parts[k] += part.tolist()
    return [np.array(p) for p in parts]

parts = dirichlet_split(train["y"].values, N_CLIENTS, ALPHA)
clients = [train.iloc[p] for p in parts]
mix = pd.DataFrame([[(c["y"] == k).sum() for k in range(5)] for c in clients],
                   columns=CLASSES, index=[f"Gateway {i+1}" for i in range(N_CLIENTS)])
mix.to_csv("results/gateway_data_mix.csv")
print("Training connections at each gateway (non-IID):")
print(mix.to_string())

# ------------------------------------------------------------------
# 3. Federated pre-processing: one-hot with the public vocabulary,
#    scaling with GLOBAL mean/std computed from per-gateway sums only
# ------------------------------------------------------------------
def encode(df):
    oh = [np.stack([(df[c].values == v) for v in VOCAB[c]], 1).astype(float) for c in CAT]
    return np.hstack(oh + [np.log1p(df[NUM].values.astype(float))])

enc_clients = [encode(c) for c in clients]
n_tot = sum(len(e) for e in enc_clients)
mean = sum(e.sum(0) for e in enc_clients) / n_tot                         # each gateway sends sums
var = sum(((e - mean) ** 2).sum(0) for e in enc_clients) / n_tot          # and squared deviations
std = np.sqrt(var) + 1e-6
Xc = [(e - mean) / std for e in enc_clients]
yc = [c["y"].values for c in clients]
X_known, y_known = (encode(test_known) - mean) / std, test_known["y"].values
X_new, y_new = (encode(test_new) - mean) / std, test_new["y"].values
print(f"\nInputs per connection: {Xc[0].shape[1]}  |  raw data stays at the gateway; only sums are shared for scaling")

# ------------------------------------------------------------------
# 4. Helpers
# ------------------------------------------------------------------
def new_model():
    return MLPClassifier(hidden_layer_sizes=HIDDEN, learning_rate_init=1e-3, batch_size=256,
                         random_state=SEED)

def train_epoch(model, X, y):
    model.partial_fit(X, y, classes=np.arange(5))       # one pass over the data = one local epoch
    return model

def get_w(m):
    return [w.copy() for w in m.coefs_ + m.intercepts_]

def set_w(m, w):
    L = len(m.coefs_)
    m.coefs_ = [x.copy() for x in w[:L]]
    m.intercepts_ = [x.copy() for x in w[L:]]

def scores(model, X, y):
    p = model.predict(X)
    return {"macro_f1": f1_score(y, p, average="macro", labels=range(5), zero_division=0),
            "accuracy": accuracy_score(y, p),
            "recall": recall_score(y, p, average=None, labels=range(5), zero_division=0)}

# ------------------------------------------------------------------
# 5. Three ways to train
# ------------------------------------------------------------------
t0 = time.time()
# (a) LOCAL ONLY: every gateway trains alone (same number of epochs)
local_models = []
for X, y in zip(Xc, yc):
    m = new_model()
    for _ in range(ROUNDS):
        train_epoch(m, X, y)
    local_models.append(m)
# (b) CENTRALISED: all data pooled in one place (upper bound, NOT privacy-preserving)
central = new_model()
Xall, yall = np.vstack(Xc), np.concatenate(yc)
for _ in range(ROUNDS):
    train_epoch(central, Xall, yall)
# (c) FEDERATED AVERAGING
global_m = new_model()
train_epoch(global_m, Xc[0][:500], yc[0][:500])                          # initialise weight shapes
sizes = np.array([len(y) for y in yc]); wts = sizes / sizes.sum()        # bigger gateway = bigger say
client_models = [copy.deepcopy(global_m) for _ in range(N_CLIENTS)]
history = []
for r in range(1, ROUNDS + 1):
    gw = get_w(global_m); updates = []
    for m, X, y in zip(client_models, Xc, yc):
        set_w(m, gw)                                                     # server -> gateway
        train_epoch(m, X, y)                                             # local training on private data
        updates.append(get_w(m))                                         # gateway -> server (weights only)
    set_w(global_m, [sum(wk * u[i] for wk, u in zip(wts, updates)) for i in range(len(gw))])  # FedAvg
    sk = scores(global_m, X_known, y_known)
    history.append({"round": r, "fedavg_macro_f1": sk["macro_f1"], "fedavg_accuracy": sk["accuracy"]})
print(f"Trained local, centralised and federated models in {time.time()-t0:.0f} s")

hist = pd.DataFrame(history); hist.to_csv("results/fedavg_rounds.csv", index=False)
n_params = sum(w.size for w in get_w(global_m))
mb_round = 2 * N_CLIENTS * n_params * 4 / 1e6

# ------------------------------------------------------------------
# 6. Results
# ------------------------------------------------------------------
rows = []
for i, m in enumerate(local_models):
    a, b = scores(m, X_known, y_known), scores(m, X_new, y_new)
    rows.append({"Model": f"Local only - Gateway {i+1}", "Macro-F1 (known attacks)": a["macro_f1"],
                 "Accuracy (known)": a["accuracy"], "Macro-F1 (KDDTest+)": b["macro_f1"],
                 "Accuracy (KDDTest+)": b["accuracy"]})
loc = pd.DataFrame(rows)
summary = [{"Model": "Local only (average of 5 gateways)", **loc.drop(columns="Model").mean().to_dict()}]
for name, m in [("Federated (FedAvg)", global_m), ("Centralised (all data pooled)", central)]:
    a, b = scores(m, X_known, y_known), scores(m, X_new, y_new)
    summary.append({"Model": name, "Macro-F1 (known attacks)": a["macro_f1"], "Accuracy (known)": a["accuracy"],
                    "Macro-F1 (KDDTest+)": b["macro_f1"], "Accuracy (KDDTest+)": b["accuracy"]})
summ = pd.DataFrame(summary)
pd.concat([loc, summ]).to_csv("results/model_comparison.csv", index=False)
pd.set_option("display.width", 200)
print("\nEach gateway alone:")
print(loc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nSummary:")
print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

rec = pd.DataFrame({"Family": CLASSES,
                    "Local (avg)": np.mean([scores(m, X_known, y_known)["recall"] for m in local_models], 0),
                    "Federated": scores(global_m, X_known, y_known)["recall"],
                    "Centralised": scores(central, X_known, y_known)["recall"]})
rec.to_csv("results/per_family_recall.csv", index=False)
print("\nRecall per attack family (known-attack test set):")
print(rec.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nCommunication: {n_params:,} parameters -> {mb_round:.2f} MB per round "
      f"({ROUNDS*mb_round:.1f} MB total) vs {Xall.nbytes/1e6:.1f} MB of raw features that never left the gateways")

# ------------------------------------------------------------------
# 7. Figures
# ------------------------------------------------------------------
ax = mix.plot(kind="bar", stacked=True, figsize=(7.5, 4.4), rot=0,
              color=["tab:gray", "tab:red", "tab:orange", "tab:blue", "tab:purple"])
ax.set_ylabel("Training connections"); ax.set_title(f"Non-IID data: what each gateway sees (Dirichlet α={ALPHA})")
plt.tight_layout(); plt.savefig("results/gateway_data_mix.png", dpi=200); plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot(hist["round"], hist["fedavg_macro_f1"], "o-", color="tab:blue", label="Federated (FedAvg)")
plt.axhline(summ.iloc[2]["Macro-F1 (known attacks)"], color="k", ls="--", label="Centralised (privacy-violating upper bound)")
plt.axhline(summ.iloc[0]["Macro-F1 (known attacks)"], color="tab:red", ls=":", label="Local only (average gateway)")
plt.xlabel("Communication round"); plt.ylabel("Macro-F1 (known-attack test set)")
plt.title("Federated learning closes the gap without sharing data"); plt.grid(alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/fedavg_convergence.png", dpi=200); plt.close()

x = np.arange(5); w = 0.27
plt.figure(figsize=(8, 4.5))
for j, (col, c) in enumerate([("Local (avg)", "tab:red"), ("Federated", "tab:blue"), ("Centralised", "k")]):
    b = plt.bar(x + (j - 1) * w, rec[col] * 100, w, label=col, color=c, alpha=0.85)
    plt.bar_label(b, fmt="%.0f", fontsize=7)
plt.xticks(x, CLASSES); plt.ylabel("Recall (%)"); plt.ylim(0, 110)
plt.title("Which traffic each approach recognises (known-attack test set)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/per_family_recall.png", dpi=200); plt.close()

plt.figure(figsize=(7.5, 4.3))
labels = [f"GW {i+1}" for i in range(N_CLIENTS)] + ["Federated", "Centralised"]
vals = list(loc["Macro-F1 (known attacks)"]) + [summ.iloc[1]["Macro-F1 (known attacks)"],
                                                 summ.iloc[2]["Macro-F1 (known attacks)"]]
b = plt.bar(labels, vals, color=["tab:red"] * N_CLIENTS + ["tab:blue", "k"])
plt.bar_label(b, fmt="%.2f", fontsize=8); plt.ylabel("Macro-F1 (known-attack test set)"); plt.ylim(0, 1.05)
plt.title("Every gateway gains by joining the federation"); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results/gateway_vs_federated.png", dpi=200); plt.close()
print("\nAll results saved in the 'results' folder.")
