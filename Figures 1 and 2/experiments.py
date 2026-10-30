from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch


# Configuration
OUTPUT_DIR = Path.cwd()
RESULTS_A_FILE = OUTPUT_DIR / "results_A.csv"
RESULTS_B_FILE = OUTPUT_DIR / "results_B.csv"
RESULTS_C_FILE = OUTPUT_DIR / "results_C.csv"

SEED = 1234
rng = np.random.default_rng(SEED)
T = 10
EPS = 1.0
TOL = 1e-10

M_VALUES = np.arange(10, 201, 10)
SETTING_B_M = 200
EPS_HAT_VALUES = np.logspace(np.log10(0.1), np.log10(20), 30)
EFFECTIVE_RANK_M = 200
EFFECTIVE_RANK_VALUES = np.linspace(1.0, EFFECTIVE_RANK_M - 1, 10)

MRO_DRO_ITERATIONS = 800
MRO_DRO_DOMAIN_GAP = 1e-3
SNR_FISTA_ITERATIONS = 60


def corrupted_sigma(m, n, effective_rank=None):
    c = m ** (-0.25)
    x = rng.standard_normal((n, m))
    y = x.copy()

    if effective_rank is None:
        y[:, 1:] += rng.standard_normal((n, m - 1))
    else:
        a = (effective_rank - 1.0) / (m - 2)
        scales = np.sqrt(np.r_[1.0, np.full(m - 2, a)])
        y[:, 1:] += rng.standard_normal((n, m - 1)) * scales

    costs = ((1.0 - c) * x[:, 0]) ** 2
    k = np.searchsorted(np.cumsum(costs), n * EPS**2, side="right")
    x[:k, 0] *= c

    z = np.hstack((x, y))
    return z.T @ z / n


def sqrt_psd(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def fit_erm(sigma, p):
    return np.linalg.solve(sigma[:p, :p], sigma[p:, :p].T).T


def clean_regret(A):
    return float(np.square(A - np.eye(A.shape[0])).sum())


def op_norm_sq(A):
    G = A.T @ A if A.shape[0] >= A.shape[1] else A @ A.T
    G = 0.5 * (G + G.T)
    return torch.linalg.eigvalsh(G).clamp_min(0.0).amax()


def kyfan_top_p_sum(H, p, jitter=TOL):
    H = 0.5 * (H + H.T)
    n = H.shape[0]
    scale = H.detach().abs().amax().clamp_min(1.0)
    ramp = torch.linspace(-1.0, 1.0, n, dtype=H.dtype, device=H.device)
    H = H + (jitter * scale) * torch.diag(ramp)
    return torch.linalg.eigvalsh(H)[-p:].sum()


def lifted_mro_objective(Sx, Sy, A, gamma, eps):
    m, p = A.shape
    residual = A @ Sx - Sy
    B = (gamma - 1.0) * torch.eye(m, dtype=A.dtype, device=A.device) - A @ A.T
    H = Sx.T @ Sx + residual.T @ torch.linalg.solve(B, residual)
    return gamma * (eps**2 - Sx.square().sum() + kyfan_top_p_sum(H, p))


def lifted_dro_objective(Sx, Sy, A, gamma, eps):
    m, _ = A.shape
    residual = A @ Sx - Sy
    B = (gamma - 1.0) * torch.eye(m, dtype=A.dtype, device=A.device) - A @ A.T
    H = residual.T @ torch.linalg.solve(B, residual)
    return gamma * (eps**2 + torch.trace(H))


def fit_mro_dro(mode, sigma, p, eps):
    S = sqrt_psd(sigma)
    Sx = torch.tensor(S[:p].copy(), dtype=torch.float64)
    Sy = torch.tensor(S[p:].copy(), dtype=torch.float64)

    A0 = fit_erm(sigma, p)
    A = torch.tensor(A0.copy(), dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(
        2.0 + np.linalg.norm(A0, 2) ** 2,
        dtype=torch.float64, requires_grad=True,
    )
    objective = lifted_mro_objective if mode == "mro" else lifted_dro_objective

    for t in range(MRO_DRO_ITERATIONS):
        A.grad = gamma.grad = None
        value = objective(Sx, Sy, A, gamma, eps)
        value.backward()

        with torch.no_grad():
            step = 1.0 / np.sqrt(t + 1.0)
            norm = torch.sqrt(A.grad.square().sum() + gamma.grad.square())
            A -= step * A.grad / norm
            gamma -= step * gamma.grad / norm
            gamma.clamp_(min=1.0 + op_norm_sq(A) + MRO_DRO_DOMAIN_GAP)

    with torch.no_grad():
        final_value = objective(Sx, Sy, A, gamma, eps)

    return A.detach().numpy(), float(gamma.detach()), float(final_value.detach())


def prox_spectral_norm_squared(B, alpha):
    U, sigma, Vt = np.linalg.svd(B, full_matrices=False)
    if sigma.size == 0 or sigma[0] <= TOL:
        return B.copy()
    csum = np.cumsum(sigma)
    sigma_ext = np.concatenate((sigma, np.zeros(1)))
    tau = csum[-1] / (sigma.size + 2.0 * alpha)
    for k in range(1, sigma.size + 1):
        candidate = csum[k - 1] / (k + 2.0 * alpha)
        if sigma_ext[k - 1] > candidate >= sigma_ext[k] - TOL:
            tau = candidate
            break
    return (U * np.minimum(sigma, tau)) @ Vt


def fit_snr(sigma, p, eps):
    A0 = fit_erm(sigma, p)
    sigma_x = 0.5 * (sigma[:p, :p] + sigma[:p, :p].T)
    sigma_yx = sigma[p:, :p]

    smoothness = 2.0 * float(np.linalg.eigvalsh(sigma_x)[-1])
    A = A0.copy()
    Y = A.copy()
    t = 1.0
    prox_alpha = eps**2 / smoothness
    for _ in range(SNR_FISTA_ITERATIONS):
        B = Y - (2.0 / smoothness) * (Y @ sigma_x - sigma_yx)
        A_next = prox_spectral_norm_squared(B, prox_alpha)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y = A_next + ((t - 1.0) / t_next) * (A_next - A)
        A, t = A_next, t_next
    return A


def evaluate(sigma, m, radius):
    A_erm = fit_erm(sigma, m)
    A_mro, gamma_mro, objective_mro = fit_mro_dro("mro", sigma, m, radius)
    A_dro, gamma_dro, objective_dro = fit_mro_dro("dro", sigma, m, radius)
    A_snr = fit_snr(sigma, m, radius)
    return {
        "erm_regret": clean_regret(A_erm),
        "mro_regret": clean_regret(A_mro),
        "mro_objective": objective_mro,
        "mro_gamma": gamma_mro,
        "dro_regret": clean_regret(A_dro),
        "dro_objective": objective_dro,
        "dro_gamma": gamma_dro,
        "snr_regret": clean_regret(A_snr),
        "A00_erm": A_erm[0, 0],
        "A00_mro": A_mro[0, 0],
        "A00_dro": A_dro[0, 0],
        "A00_snr": A_snr[0, 0],
    }


def show_progress(label, row, start):
    elapsed = time.perf_counter() - start
    print(
        f"{label} "
        f"ERM={row['erm_regret']:.4f} "
        f"MRO={row['mro_regret']:.4f} "
        f"DRO={row['dro_regret']:.4f} "
        f"SNR={row['snr_regret']:.4f} "
        f"({elapsed / 60:.1f} min)", flush=True,
    )


# Setting A
rows = []
start = time.perf_counter()
for m in M_VALUES:
    n = 10 * m**2
    for t in range(T):
        sigma = corrupted_sigma(m, n)
        row = {"m": m, "n": n, "t": t, "seed": SEED}
        row.update(evaluate(sigma, m, EPS))
        rows.append(row)
        show_progress(f"A: m={m}, t={t}", row, start)
pd.DataFrame(rows).to_csv(RESULTS_A_FILE, index=False)
print(f"Saved {RESULTS_A_FILE}")


# Setting B
rows = []
m = SETTING_B_M
n = 10 * m**2
start = time.perf_counter()
for eps_hat in EPS_HAT_VALUES:
    for t in range(T):
        sigma = corrupted_sigma(m, n)
        row = {"m": m, "n": n, "t": t, "seed": SEED, "eps_hat": eps_hat}
        row.update(evaluate(sigma, m, eps_hat))
        rows.append(row)
        show_progress(f"B: rho={eps_hat:.4g}, t={t}", row, start)
pd.DataFrame(rows).to_csv(RESULTS_B_FILE, index=False)
print(f"Saved {RESULTS_B_FILE}")


# Setting C
rows = []
m = EFFECTIVE_RANK_M
n = 10 * m**2
start = time.perf_counter()
for r in EFFECTIVE_RANK_VALUES:
    for t in range(T):
        sigma = corrupted_sigma(m, n, effective_rank=r)
        row = {
            "m": m, "n": n, "t": t, "seed": SEED,
            "effective_rank": float(r), "eps_hat": EPS,
        }
        row.update(evaluate(sigma, m, EPS))
        rows.append(row)
        show_progress(f"C: r={r:.4g}, t={t}", row, start)
pd.DataFrame(rows).to_csv(RESULTS_C_FILE, index=False)
print(f"Saved {RESULTS_C_FILE}")