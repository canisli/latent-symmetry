import argparse
import numpy as np
import matplotlib.pyplot as plt
import energyflow as ef

from data.kp_dataset import compute_kps


def generate_kp_values(
    num_events: int,
    num_particles: int,
    edges_list,
    measure: str,
    coords: str,
    seed: int,
    beta: float = 2.0,
    kappa: float = 1.0,
    normed: bool = False,
):
    """Generate KP/EFP values from synthetic events."""
    np.random.seed(seed)
    X = ef.gen_random_events_mcom(num_events, num_particles, dim=4).astype(np.float32)
    kp_vals = compute_kps(X, edges_list, measure=measure, coords=coords, beta=beta, kappa=kappa, normed=normed)
    return kp_vals.reshape(-1)


def main():
    parser = argparse.ArgumentParser(description="Plot histogram of KP/EFP values")
    parser.add_argument("--num-events", type=int, default=2_000, help="Number of events")
    parser.add_argument("--num-particles", type=int, default=128, help="Particles per event")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Target type arguments
    parser.add_argument("--target-type", type=str, default="kinematic",
                        choices=["kinematic", "efp"],
                        help="Target type: kinematic (Lorentz invariant) or efp (non-invariant)")
    parser.add_argument("--beta", type=float, default=2.0, help="EFP beta parameter")
    parser.add_argument("--kappa", type=float, default=1.0, help="EFP kappa parameter")
    parser.add_argument("--normed", action="store_true",
                        help="Normalize energies so sum(z_i)=1 (default: False)")
    args = parser.parse_args()

    edges_list = [[(0, 1), (0, 2), (0, 3)]]
    
    # Determine measure based on target type
    if args.target_type == "kinematic":
        measure = "kinematic"
        target_label = "Kinematic polynomial"
        invariance_note = "Lorentz invariant"
    else:  # efp
        measure = "eeefm"
        normed_str = "normed" if args.normed else "unnormed"
        target_label = f"EFP (eeefm, β={args.beta}, κ={args.kappa}, {normed_str})"
        invariance_note = "NOT Lorentz invariant"
    
    print(f"Target type: {args.target_type}")
    print(f"Measure: {measure}")
    if args.target_type == "efp":
        print(f"Normed: {args.normed}")
    print(f"Invariance: {invariance_note}")
    print()
    
    values_raw = generate_kp_values(
        num_events=args.num_events,
        num_particles=args.num_particles,
        edges_list=edges_list,
        measure=measure,
        coords="epxpypz",
        seed=args.seed,
        beta=args.beta,
        kappa=args.kappa,
        normed=args.normed,
    )

    # Apply log1p transformation (matching labels used in train.py)
    values = np.log1p(values_raw)

    # Basic stats on log-transformed values
    print(f"Count: {values.size}")
    print(f"Min / Max: {values.min():.4e} / {values.max():.4e}")
    print(f"Mean / Std: {values.mean():.4e} / {values.std():.4e}")
    print(f"Median: {np.median(values):.4e}")
    print(f"1st / 99th pct: {np.percentile(values, 1):.4e} / {np.percentile(values, 99):.4e}")
    
    # Also show raw (pre-log1p) stats
    print()
    print("Raw (pre-log1p) stats:")
    print(f"Min / Max: {values_raw.min():.4e} / {values_raw.max():.4e}")
    print(f"Mean / Std: {values_raw.mean():.4e} / {values_raw.std():.4e}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(values, bins=args.bins, edgecolor="k", alpha=0.7)
    ax.set_xlabel("log1p(value)")
    ax.set_ylabel("Count")
    ax.set_title(f"{target_label} value distribution (log1p-transformed)\n{invariance_note}")
    ax.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
