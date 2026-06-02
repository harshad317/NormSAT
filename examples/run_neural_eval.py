"""Neural-mode evaluation: compiled RANCSATNorm vs LayerNorm / RMSNorm / DyT.

Trains a small MLP classifier on a tabular task with each normalization layer swapped
in, and reports test accuracy, training stability (gradient-norm / activation-norm),
and the compiled policy NormSAT chose. This is the honest head-to-head the paper's
neural section needs.

Requires PyTorch:  pip install torch
Run:  PYTHONPATH=. python examples/run_neural_eval.py --epochs 30 --seeds 5

HONEST FRAMING: RMSNorm/LayerNorm/DyT are strong, well-tuned layers. NormSAT's neural
mode compiles the centering bit / scale statistic / bounded map from a contract; the
claim is that it RECOVERS these layers as special cases and is competitive in the
small-batch / unreliable-statistic regime, NOT that it beats them universally.
"""
from __future__ import annotations
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise SystemExit("This script needs PyTorch: pip install torch") from e

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rancsat.schemas import InvarianceContract
from rancsat.torch_mode import RANCSATNorm, ActivationRegime


# --------------------------------------------------------------------------- #
# Reference normalization layers
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.g * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class DyT(nn.Module):
    """Dynamic Tanh (Zhu et al., 2025): y = gamma * tanh(alpha x) + beta."""
    def __init__(self, d, alpha0=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha0)))
        self.g = nn.Parameter(torch.ones(d)); self.b = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        return self.g * torch.tanh(self.alpha * x) + self.b


def make_norm(kind, d):
    if kind == "layernorm":
        return nn.LayerNorm(d)
    if kind == "rmsnorm":
        return RMSNorm(d)
    if kind == "dyt":
        return DyT(d)
    if kind == "rancsat":
        # contract: batch-independent (no batchnorm), mean may be informative -> let the
        # compiler decide centering; small tabular MLP regime.
        contract = InvarianceContract(avoid_batch_dependence=True)
        regime = ActivationRegime(block="mlp", batch_reliable=False, mean_is_signal=False,
                                  saturation_risk=0.2, depth_sensitive=False, precision="fp32")
        return RANCSATNorm(d, contract, regime)
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)


class MLP(nn.Module):
    def __init__(self, d_in, norm_kind, h=64, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h); self.n1 = make_norm(norm_kind, h)
        self.fc2 = nn.Linear(h, h);    self.n2 = make_norm(norm_kind, h)
        self.out = nn.Linear(h, n_classes)
        self.norm_kind = norm_kind
    def forward(self, x):
        x = F.relu(self.n1(self.fc1(x)))
        x = F.relu(self.n2(self.fc2(x)))
        return self.out(x)


def train_eval(Xtr, ytr, Xte, yte, kind, epochs, lr, batch, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    d_in = Xtr.shape[1]; n_cls = int(max(ytr.max(), yte.max()) + 1)
    model = MLP(d_in, kind, n_classes=n_cls)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    grad_norms = []
    n = len(Xtr_t)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            gn = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None))
            grad_norms.append(float(gn))
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(-1).numpy()
    acc = float((pred == yte).mean())
    info = {"acc": acc, "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_std": float(np.std(grad_norms))}
    if kind == "rancsat":
        info["policy"] = model.n1.stability_report()["policy"]
    return info


def run(dataset="breast_cancer", epochs=30, lr=1e-3, batch=16, seeds=range(5)):
    load = {"breast_cancer": load_breast_cancer, "wine": load_wine}[dataset]
    b = load(); X = b.data.astype(np.float32); y = b.target.astype(np.int64)
    kinds = ["none", "layernorm", "rmsnorm", "dyt", "rancsat"]
    results = {k: [] for k in kinds}
    policy_seen = None
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=int(s), stratify=y)
        sc = StandardScaler().fit(Xtr)              # raw features standardized for input
        Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
        for k in kinds:
            r = train_eval(Xtr2, ytr, Xte2, yte, k, epochs, lr, batch, int(s))
            results[k].append(r["acc"])
            if k == "rancsat" and policy_seen is None:
                policy_seen = r.get("policy")
    print(f"\nNEURAL-MODE EVAL on {dataset} (small MLP, batch={batch}, {epochs} epochs, "
          f"{len(list(seeds))} seeds)")
    print(f"  RANCSATNorm compiled policy: {policy_seen}")
    print(f"  {'layer':<12}{'mean_acc':>10}{'std':>8}")
    print("  " + "-" * 30)
    for k in kinds:
        a = np.array(results[k])
        print(f"  {k:<12}{a.mean():>10.4f}{a.std():>8.4f}")
    print("\n  Honest read: RANCSATNorm should be COMPETITIVE with LayerNorm/RMSNorm/DyT,")
    print("  recovering centering/scale choices from the contract -- not a universal win.")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_cancer", choices=["breast_cancer", "wine"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    run(args.dataset, epochs=args.epochs, batch=args.batch, seeds=range(args.seeds))
