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
    kp_scale: float | None,
    seed: int,
):
    """Generate KP values from synthetic events."""
    np.random.seed(seed)
    X = ef.gen_random_events_mcom(num_events, num_particles, dim=4).astype(np.float32)
    kp_vals = compute_kps(X, edges_list, measure=measure, coords=coords)
    if kp_scale is not None and kp_scale != 0:
        kp_vals = kp_vals / float(kp_scale)
    return kp_vals.reshape(-1)


def main():
    parser = argparse.ArgumentParser(description="Plot histogram of KP values")
    parser.add_argument("--num-events", type=int, default=2_000, help="Number of events")
    parser.add_argument("--num-particles", type=int, default=128, help="Particles per event")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    parser.add_argument("--kp-scale", type=float, default=None,
                        help="Optional scale to divide KP values by (default: None)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-path", type=str, default="kp_histogram.png",
                        help="Path to save histogram PNG")
    parser.add_argument("--show", action="store_true", help="Show the plot window")
    args = parser.parse_args()

    edges_list = [[(0, 1), (0, 2), (0, 3)]]
    kp_values = generate_kp_values(
        num_events=args.num_events,
        num_particles=args.num_particles,
        edges_list=edges_list,
        measure="kinematic",
        coords="epxpypz",
        kp_scale=args.kp_scale,
        seed=args.seed,
    )

    # Basic stats
    print(f"Count: {kp_values.size}")
    print(f"Min / Max: {kp_values.min():.4e} / {kp_values.max():.4e}")
    print(f"Mean / Std: {kp_values.mean():.4e} / {kp_values.std():.4e}")
    print(f"Median: {np.median(kp_values):.4e}")
    print(f"1st / 99th pct: {np.percentile(kp_values, 1):.4e} / {np.percentile(kp_values, 99):.4e}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(kp_values, bins=args.bins, edgecolor="k", alpha=0.7)
    label = "KP value"
    if args.kp_scale is not None and args.kp_scale != 0:
        label += f" (scaled by {args.kp_scale:g})"
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title("Kinematic polynomial value distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.save_path, dpi=150)
    print(f"Saved histogram to {args.save_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

