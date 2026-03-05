# SpaRCe_Adiac.py
# SpaRCe (PyTorch) – single reservoir architecture + SpaRCe readout, adapted to UCR Adiac dataset (176 time steps)

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Adiac (UCR)
    adiac_train_path: str = "Adiac/Adiac_TRAIN.txt"
    adiac_test_path: str = "Adiac/Adiac_TEST.txt"
    T: int = 176
    input_dim: int = 1
    n_classes: int = 37

    # Reservoir
    N: int = 1000
    dt: float = 0.01
    tau_m: float = 0.03
    tau_M: float = 2.0
    diluition: float = 0.99  # keep spelling to match original code style

    # Precompute / eval
    batch_size_states: int = 256
    batch_size_eval: int = 512

    # Training
    lr_wout: float = 0.002
    lr_theta_factor: float = 0.1
    adam_eps: float = 1e-7

    # Data normalization (per-sample z-normalization)
    z_norm_per_sample: bool = True


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset loader (UCR .txt)
# -----------------------------
class UCRTxtDataset(Dataset):
    """
    UCR .txt format: each row is:
        label  x_1 x_2 ... x_T
    For Adiac, T=176. (Label is 1..37 in the raw file.)
    """
    def __init__(self, path: str, T: int, z_norm_per_sample: bool = True):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        data = torch.tensor(
            # loadtxt is fine here (dataset is small-ish); keeps script dependency-free
            __import__("numpy").loadtxt(path),
            dtype=torch.float32,
        )

        if data.ndim != 2 or data.shape[1] != (1 + T):
            raise ValueError(f"{path}: expected shape [N, {1+T}] but got {tuple(data.shape)}")

        y_raw = data[:, 0]
        X = data[:, 1:]  # [N,T]

        # Map labels to 0..C-1 (UCR labels are often 1..C)
        uniq = torch.unique(y_raw).cpu().tolist()
        uniq_sorted = sorted(float(u) for u in uniq)
        mapping = {lab: i for i, lab in enumerate(uniq_sorted)}
        y = torch.tensor([mapping[float(v)] for v in y_raw.cpu().tolist()], dtype=torch.int64)

        if z_norm_per_sample:
            mu = X.mean(dim=1, keepdim=True)
            sd = X.std(dim=1, keepdim=True).clamp_min(1e-6)
            X = (X - mu) / sd

        # store as [N,1,T] (input_dim=1)
        self.X = X.unsqueeze(1).contiguous()
        self.y = y.contiguous()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_adiac(cfg: Cfg):
    ds_train = UCRTxtDataset(cfg.adiac_train_path, T=cfg.T, z_norm_per_sample=cfg.z_norm_per_sample)
    ds_test = UCRTxtDataset(cfg.adiac_test_path, T=cfg.T, z_norm_per_sample=cfg.z_norm_per_sample)

    # shuffle=False because we precompute in fixed order (like your MNIST version)
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size_states, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size_states, shuffle=False, num_workers=0)

    return train_loader, test_loader


# -----------------------------
# ESN helpers (copied style from your MNIST version)
# -----------------------------
def alpha_pho(dt: float, tau_m: float, tau_M: float):
    alpha = dt / (2.0 * tau_m)
    pho = 1.0 - 2.0 * tau_m / tau_M
    return float(alpha), float(pho)


@torch.no_grad()
def spectral_radius(W: torch.Tensor) -> float:
    ev = torch.linalg.eigvals(W.detach().cpu())
    return float(max(ev.abs().max().item(), 1e-8))


@torch.no_grad()
def make_sparse_W(N: int, diluition: float, pho: float, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = (torch.rand((N, N), generator=g) * 2.0 - 1.0).to(torch.float32)

    D = (torch.rand((N, N), generator=g) > diluition).to(torch.float32)
    W = W * D

    sr = spectral_radius(W)
    W = pho * W / sr
    return W.to(device)


class ESN1:
    def __init__(self, cfg: Cfg, seed: int, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.alpha, pho = alpha_pho(cfg.dt, cfg.tau_m, cfg.tau_M)
        self.W = make_sparse_W(cfg.N, cfg.diluition, pho, seed + 1, device)

        g = torch.Generator(device="cpu").manual_seed(seed + 2)
        W_in = torch.randn((cfg.N, cfg.input_dim), generator=g).to(torch.float32)
        self.W_in = (0.1 * W_in.t()).to(device)  # (input_dim, N)

    @torch.no_grad()
    def forward_states(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B,input_dim,T]
        returns: [B,N,T]
        """
        if X.ndim != 3 or X.shape[1] != self.cfg.input_dim or X.shape[2] != self.cfg.T:
            raise ValueError(f"Expected X [B,{self.cfg.input_dim},{self.cfg.T}], got {tuple(X.shape)}")

        B = X.shape[0]
        v = torch.zeros((B, self.cfg.N), device=self.device, dtype=torch.float32)

        states = []
        for t in range(self.cfg.T):
            u = X[:, :, t]  # [B,input_dim]
            v = (1.0 - self.alpha) * v + self.alpha * torch.tanh(v @ self.W + u @ self.W_in)
            states.append(v)

        return torch.stack(states, dim=2)  # [B,N,T]


# -----------------------------
# SpaRCe readout
# -----------------------------
class SpaRCeReadout(nn.Module):
    def __init__(self, D: int, N_scale: int, n_classes: int, theta_g_flat: torch.Tensor, seed: int):
        super().__init__()

        g = torch.Generator(device="cpu").manual_seed(seed + 100)
        self.W_out = nn.Parameter(torch.rand((D, n_classes), generator=g).to(torch.float32) / float(N_scale))

        g = torch.Generator(device="cpu").manual_seed(seed + 101)
        self.theta_i = nn.Parameter(torch.randn((1, D), generator=g).to(torch.float32) / float(N_scale))

        # fixed theta_g from train set percentile
        self.register_buffer("theta_g", theta_g_flat.reshape(1, D).to(torch.float32))

    def forward(self, s_flat: torch.Tensor) -> torch.Tensor:
        thr = self.theta_g + self.theta_i
        s_sparse = torch.sign(s_flat) * F.relu(s_flat.abs() - thr)
        return s_sparse @ self.W_out


# -----------------------------
# Precompute states + theta_g
# -----------------------------
@torch.no_grad()
def precompute_states(reservoir: ESN1, loader: DataLoader, device: torch.device):
    y_all = []
    states_all = []

    for xb, yb in loader:
        xb = xb.to(device)  # [B,1,176]
        yb = yb.to(device)

        st = reservoir.forward_states(xb)  # [B,N,T]

        states_all.append(st.cpu())
        y_all.append(yb.cpu())

    S = torch.cat(states_all, dim=0)  # [Nsam,N,T]
    y = torch.cat(y_all, dim=0)      # [Nsam]

    # flatten reservoir time dynamics: V~ = concat_t V(t)  => [Nsam, N*T]
    Xflat = S.permute(0, 1, 2).reshape(S.shape[0], -1).contiguous()
    return Xflat, y, S


@torch.no_grad()
def theta_g_from_train(Str: torch.Tensor, Pn: float) -> torch.Tensor:
    """
    Str: [Nsam,N,T] states
    returns theta_g_flat: [N*T]
    """
    Sflat = Str.reshape(Str.shape[0], -1).abs()  # [Nsam, N*T]
    q = float(Pn) / 100.0
    theta_g = torch.quantile(Sflat, q=q, dim=0)  # [N*T]
    return theta_g


# -----------------------------
# Training utilities
# -----------------------------
def one_hot(y: torch.Tensor, n_classes: int, device: torch.device) -> torch.Tensor:
    oh = torch.zeros((y.shape[0], n_classes), device=device, dtype=torch.float32)
    oh.scatter_(1, y.view(-1, 1), 1.0)
    return oh


@torch.no_grad()
def accuracy(readout: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device, bs: int) -> float:
    readout.eval()
    correct = 0
    total = 0
    for i in range(0, X.shape[0], bs):
        xb = X[i : i + bs].to(device)
        yb = y[i : i + bs].to(device)
        pred = readout(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return float(correct / max(total, 1))


def train_iterations(
    readout: SpaRCeReadout,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    n_episodes: int,
    batch_size_train: int,
    n_check: int,
    cfg: Cfg,
    device: torch.device,
):
    # Same loss as your MNIST version (sigmoid cross-entropy on one-hot)
    # tf.losses.sigmoid_cross_entropy(y_true, y_logits)
    loss_fn = lambda logits, yoh: F.binary_cross_entropy_with_logits(logits, yoh)

    opt = torch.optim.Adam(
        [
            {"params": [readout.W_out], "lr": cfg.lr_wout},
            {"params": [readout.theta_i], "lr": cfg.lr_wout * cfg.lr_theta_factor},
        ],
        eps=cfg.adam_eps,
    )

    n = int(ytr.shape[0])
    check_every = max(n_episodes // max(int(n_check), 1), 1)

    for it in range(1, n_episodes + 1):
        idx = torch.randint(0, n, (batch_size_train,))
        xb = Xtr[idx].to(device)
        yb = ytr[idx].to(device)
        yoh = one_hot(yb, cfg.n_classes, device)

        readout.train()
        opt.zero_grad(set_to_none=True)
        logits = readout(xb)
        loss = loss_fn(logits, yoh)
        loss.backward()
        opt.step()

        if it % check_every == 0 or it == 1:
            tr_acc = accuracy(readout, Xtr, ytr, device, cfg.batch_size_eval)
            print(f"    iter {it:>7d}/{n_episodes} | train_acc={tr_acc:.4f}")


@torch.no_grad()
def predict_logits(readout: nn.Module, X: torch.Tensor, device: torch.device, bs: int) -> torch.Tensor:
    readout.eval()
    outs = []
    for i in range(0, X.shape[0], bs):
        outs.append(readout(X[i : i + bs].to(device)).cpu())
    return torch.cat(outs, dim=0)


# -----------------------------
# Main experiment (Adiac)
# -----------------------------
def run_adiac():
    # --- settings you typically change ---
    Pn_list = [85.0, 90.0, 95.0]   # starting sparsity percentile(s)
    runs_to_average = 5            # Adiac is smaller; 5 is a reasonable start
    n_episodes = 200_000           # you can raise/lower
    batch_size_train = 20
    n_check = 50
    # ------------------------------------

    cfg = Cfg()
    device = torch.device(cfg.device)

    print("[Adiac] loading...")
    train_loader, test_loader = load_adiac(cfg)

    results = []

    for Pn in Pn_list:
        print("\n==============================")
        print(f"Pn = {Pn}")
        print("==============================")

        test_logits_sum = None
        test_y_ref = None
        test_accs = []

        for r in range(runs_to_average):
            seed = 123 + r
            set_seed(seed)

            reservoir = ESN1(cfg, seed=seed, device=device)

            print(f"  Run {r + 1}/{runs_to_average} | seed={seed}")
            print("    precompute TRAIN states...")
            Xtr, ytr, Str = precompute_states(reservoir, train_loader, device)

            print("    compute theta_g (full train)...")
            theta_g = theta_g_from_train(Str, Pn)

            print("    precompute TEST states...")
            Xte, yte, _ = precompute_states(reservoir, test_loader, device)
            if test_y_ref is None:
                test_y_ref = yte

            D = int(Xtr.shape[1])  # N*T
            readout = SpaRCeReadout(D=D, N_scale=cfg.N, n_classes=cfg.n_classes, theta_g_flat=theta_g, seed=seed).to(device)

            print("    train (iterations)...")
            train_iterations(
                readout=readout,
                Xtr=Xtr,
                ytr=ytr,
                n_episodes=n_episodes,
                batch_size_train=batch_size_train,
                n_check=n_check,
                cfg=cfg,
                device=device,
            )

            te_acc = accuracy(readout, Xte, yte, device, cfg.batch_size_eval)
            test_accs.append(te_acc)
            print(f"    test_acc={te_acc:.4f}")

            logits = predict_logits(readout, Xte, device, cfg.batch_size_eval)
            test_logits_sum = logits if test_logits_sum is None else (test_logits_sum + logits)

        mean_test = sum(test_accs) / len(test_accs)
        ensemble_logits = test_logits_sum / float(runs_to_average)
        ensemble_acc = float((ensemble_logits.argmax(dim=1) == test_y_ref).to(torch.float32).mean().item())

        print("\n  ---- Summary ----")
        print(f"  test accs: {[f'{a:.4f}' for a in test_accs]}")
        print(f"  mean test acc: {mean_test:.4f}")
        print(f"  ensemble(avg logits) test acc: {ensemble_acc:.4f}")

        results.append((Pn, mean_test, ensemble_acc))

    print("\n==============================")
    print("All Pn results:")
    for pn, mean_test, ens in results:
        print(f"  Pn={pn:>5} | mean_test={mean_test:.4f} | ensemble_test={ens:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    run_adiac()