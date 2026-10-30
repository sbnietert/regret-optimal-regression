from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import torch


# Configuration
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "institutions" / "agg_1_hour"
MAX_M = 268
COVERAGE = 0.99
RESULTS_FILE = OUTPUT_DIR / "results_cesnet.csv"
GEOMETRY_FILE = OUTPUT_DIR / "clean_geometry_cesnet.csv"
TOL = 1e-10
M_VALUES = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 268]
GAIN = 0.5

MRO_DRO_ITERATIONS = 10000
MRO_DRO_DOMAIN_GAP = 1e-3
SNR_FISTA_ITERATIONS = 10000


def load_data(path=DATA_DIR):
    # Read the publisher's hourly CSVs; keep one fixed cohort at every m.
    files = sorted(path.glob("*.csv"), key=lambda file: int(file.stem))
    if not files:
        raise FileNotFoundError(f"No hourly institution CSVs in {path}")
    panels = {}
    for file in files:
        frame = pd.read_csv(file, usecols=["id_time", "n_packets", "n_bytes"])
        frame = frame.set_index("id_time").sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate hourly time ID in {file.name}")
        panels[int(file.stem)] = frame
    observed_times = set().union(*(frame.index for frame in panels.values()))
    eligible = []
    for ident, frame in panels.items():
        values = frame.to_numpy()
        good = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
        if (good.sum() / len(observed_times) >= COVERAGE
                and frame.n_packets.std(ddof=0) > 0
                and frame.n_bytes.std(ddof=0) > 0):
            eligible.append(ident)
    if len(eligible) < MAX_M:
        raise ValueError(f"Only {len(eligible)} qualifying institutions; need {MAX_M}")
    ids = eligible[:MAX_M]
    full = pd.concat({ident: panels[ident] for ident in ids}, axis=1, join="inner")
    full = full.dropna().sort_index()
    values = full.to_numpy()
    full = full.loc[np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)]
    raw_x = full.xs("n_packets", level=1, axis=1).to_numpy(float)[:, :MAX_M // 2]
    raw_y = full.xs("n_bytes", level=1, axis=1).to_numpy(float)
    mx, sx = raw_x.mean(axis=0), raw_x.std(axis=0)
    my, sy = raw_y.mean(axis=0), raw_y.std(axis=0)
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("A selected counter is constant on the common timestamps.")
    x, y = (raw_x - mx) / sx, (raw_y - my) / sy
    print(f"Loaded {len(files)} institutions; {len(eligible)} qualifying; "
          f"using {len(ids)} institutions and {len(full)} common hours.", flush=True)
    return x, y, mx[0] / sx[0]


def moments(x, y):
    z = np.hstack((x, y))
    return z.T @ z / len(z)


def corrupted_sigma(x, y, m, gain, raw_mean_over_std):
    q = m // 2
    X = np.column_stack((np.ones(len(x)), x[:, :q]))
    target = moments(X, y[:, :m])
    faulty = X.copy()
    # Multiply RAW counts by c, then apply the frozen clean normalization.
    # Keep the source mean shift. Column 0 is the constant, column 1 faulty.
    faulty[:, 1] = gain * X[:, 1] + (gain - 1) * raw_mean_over_std
    eps = float(np.sqrt(np.mean((faulty[:, 1] - X[:, 1]) ** 2)))
    return moments(faulty, y[:, :m]), target, eps



def sqrt_psd(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def fit_erm(sigma, p):
    return np.linalg.solve(sigma[:p, :p], sigma[p:, :p].T).T


def clean_regret(A, target_sigma, p):
    clean = fit_erm(target_sigma, p)
    diff = A - clean
    return float(np.sum((diff @ target_sigma[:p, :p]) * diff))


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


def evaluate(sigma, p, radius, target_sigma):
    A_erm = fit_erm(sigma, p)
    A_mro, gamma_mro, objective_mro = fit_mro_dro("mro", sigma, p, radius)
    A_dro, gamma_dro, objective_dro = fit_mro_dro("dro", sigma, p, radius)
    A_snr = fit_snr(sigma, p, radius)
    return {
        "erm_regret": clean_regret(A_erm, target_sigma, p),
        "mro_regret": clean_regret(A_mro, target_sigma, p),
        "mro_objective": objective_mro,
        "mro_gamma": gamma_mro,
        "dro_regret": clean_regret(A_dro, target_sigma, p),
        "dro_objective": objective_dro,
        "dro_gamma": gamma_dro,
        "snr_regret": clean_regret(A_snr, target_sigma, p),
        "A01_erm": A_erm[0, 1],
        "A01_mro": A_mro[0, 1],
        "A01_dro": A_dro[0, 1],
        "A01_snr": A_snr[0, 1],
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

def target_geometry(x, y):
    rows = []
    for q in range(1, x.shape[1] + 1):
        m, p = 2 * q, q + 1
        X = np.column_stack((np.ones(len(x)), x[:, :q]))
        sigma = moments(X, y[:, :m])
        A = fit_erm(sigma, p)
        sigma_x = sigma[:p, :p]
        sigma_e = sigma[p:, p:] - A @ sigma[:p, p:]
        sigma_e = (sigma_e + sigma_e.T) / 2
        lambda_min = np.linalg.eigvalsh(sigma_x)[0]
        ev_e = np.linalg.eigvalsh(sigma_e)
        kappa = 1 + np.linalg.eigvalsh(A @ A.T + sigma_e / lambda_min)[-1]
        rows.append({
            "m": m, "q": q, "p": p, "n": len(x),
            "residual_effective_rank": float(np.trace(sigma_e) / ev_e[-1]),
            "kappa": float(kappa),
        })
    return pd.DataFrame(rows)


def main():
    global MRO_DRO_ITERATIONS, SNR_FISTA_ITERATIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR,
                        help="Directory of publisher hourly institution CSVs")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dimensions", type=int, nargs="+", default=M_VALUES)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=MRO_DRO_ITERATIONS)
    parser.add_argument("--snr-iterations", type=int, default=SNR_FISTA_ITERATIONS)
    args = parser.parse_args()
    if any(m < 2 or m > 268 or m % 2 for m in args.dimensions):
        parser.error("Dimensions must be even integers from 2 through 268.")
    if min(args.iterations, args.snr_iterations) < 1:
        parser.error("Iteration counts must be positive.")
    MRO_DRO_ITERATIONS, SNR_FISTA_ITERATIONS = args.iterations, args.snr_iterations
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, y, ratio = load_data(args.data)
    geometry = target_geometry(x, y)
    geometry.to_csv(args.output_dir / GEOMETRY_FILE.name, index=False)
    if args.geometry_only:
        return

    rows = []
    start = time.perf_counter()
    results_file = args.output_dir / RESULTS_FILE.name
    for m in args.dimensions:
        q, p = m // 2, m // 2 + 1
        sigma, target_sigma, radius = corrupted_sigma(x, y, m, GAIN, ratio)
        row = {"m": m, "q": q, "p": p, "n": len(x), "gain": GAIN, "eps": radius,
               "robust_iterations": MRO_DRO_ITERATIONS, "snr_iterations": SNR_FISTA_ITERATIONS}
        row.update(evaluate(sigma, p, radius, target_sigma))
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_file, index=False)
        show_progress(f"CESNET: m={m}, q={q}, c={GAIN:g}", row, start)
    print(f"Saved {results_file}")


if __name__ == "__main__":
    main()
