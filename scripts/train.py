#!/usr/bin/env python3

from latsym.models import MLP, build_model
from latsym.seeds import derive_seed, set_model_seed, set_global_seed
from latsym.tasks import create_dataloaders
from latsym.train import train_loop, create_scheduler, plot_loss_curves
from latsym.eval import plot_regression_surface, plot_run_summary
from latsym.symmetry_penalty import (
    create_symmetry_penalty,
    PeriodicPCAOrbitVariancePenalty,
)
from latsym.metrics import get_metric
from latsym.metrics.q_metric import compute_oracle_Q, plot_Q_vs_layer, plot_Q_h_vs_layer
from latsym.metrics.rsl_metric import plot_rsl_vs_layer
from latsym.metrics.sl_metric import plot_sl_vs_layer
from latsym.metrics.mi_metric import compute_oracle_MI

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys
import shlex
import logging

log = logging.getLogger(__name__)


def compute_and_plot_metrics(
    model: nn.Module,
    full_dataset,
    cfg: DictConfig,
    output_dir: Path,
    device: torch.device,
    run_name: str = None,
    field_name: str = None,
    sym_penalty_type: str = None,
    sym_layers: list = None,
    lambda_sym: float = 0.0,
):
    """
    Compute all metrics and save plots.
    
    Args:
        model: Trained model.
        full_dataset: Full dataset with get_numpy() method.
        cfg: Hydra config with metrics section.
        output_dir: Directory to save plots and metrics.
        device: Torch device.
        run_name: Optional run name for plot titles.
        field_name: Name of the scalar field used for training.
        sym_penalty_type: Type of symmetry penalty used during training.
        sym_layers: List of layers penalized during training.
        lambda_sym: Lambda value for symmetry penalty.
    
    Returns:
        Dictionary with all metric values.
    """
    # Get full dataset as tensor
    X, y = full_dataset.get_numpy()
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y.ndim == 1 else torch.tensor(y, dtype=torch.float32)
    
    # Compute Q metric
    metric_cfg = OmegaConf.to_container(cfg.metrics.get("Q", {}), resolve=True)
    metric = get_metric("Q", **metric_cfg)
    Q_values = metric.compute(model, X, device=device)
    
    # Compute Q_h metric (raw activations, no PCA)
    metric_h_cfg = OmegaConf.to_container(cfg.metrics.get("Q_h", {}), resolve=True)
    metric_h = get_metric("Q_h", **metric_h_cfg)
    Q_h_values = metric_h.compute(model, X, device=device)
    
    # Compute RSL metric (relative symmetry loss)
    rsl_cfg = OmegaConf.to_container(cfg.metrics.get("RSL", {}), resolve=True)
    rsl_metric = get_metric("RSL", **rsl_cfg)
    RSL_values = rsl_metric.compute(model, X, device=device)
    
    # Compute SL metric (raw symmetry loss)
    sl_cfg = OmegaConf.to_container(cfg.metrics.get("SL", {}), resolve=True)
    sl_metric = get_metric("SL", **sl_cfg)
    SL_values = sl_metric.compute(model, X, device=device)
    
    # Compute oracle Q
    oracle_Q = compute_oracle_Q(X, y, full_dataset.scalar_field_fn, n_rotations=32, device=device)
    
    # Log metrics
    log.info(f"\n{'='*40}")
    log.info("Metric Results")
    log.info(f"{'='*40}")
    log.info(f"Oracle Q = {oracle_Q:.4f}")
    log.info(f"Q values by layer:")
    for layer, val in Q_values.items():
        log.info(f"  {layer}: Q = {val:.4f}")
    log.info(f"Q_h values by layer:")
    for layer, val in Q_h_values.items():
        log.info(f"  {layer}: Q_h = {val:.4f}")
    log.info(f"RSL values by layer:")
    for layer, val in RSL_values.items():
        log.info(f"  {layer}: RSL = {val:.4f}")
    log.info(f"SL values by layer:")
    for layer, val in SL_values.items():
        log.info(f"  {layer}: SL = {val:.6f}")
    
    # Save metric values
    all_metric_values = {
        "Q": Q_values,
        "Q_h": Q_h_values,
        "RSL": RSL_values,
        "SL": SL_values,
        "oracle_Q": oracle_Q,
    }
    with open(output_dir / 'metric_values.json', 'w') as f:
        json.dump(all_metric_values, f, indent=2)
    
    # Save individual metric plots
    plot_Q_vs_layer(
        Q_values, output_dir / 'Q_vs_layer.png', oracle_Q=oracle_Q, run_name=run_name,
        field_name=field_name, sym_penalty_type=sym_penalty_type, sym_layers=sym_layers, lambda_sym=lambda_sym
    )
    plot_Q_h_vs_layer(
        Q_h_values, output_dir / 'Q_h_vs_layer.png', run_name=run_name,
        field_name=field_name, sym_penalty_type=sym_penalty_type, sym_layers=sym_layers, lambda_sym=lambda_sym
    )
    plot_rsl_vs_layer(
        RSL_values, output_dir / 'RSL_vs_layer.png', run_name=run_name,
        field_name=field_name, sym_penalty_type=sym_penalty_type, sym_layers=sym_layers, lambda_sym=lambda_sym
    )
    plot_sl_vs_layer(
        SL_values, output_dir / 'SL_vs_layer.png', run_name=run_name,
        field_name=field_name, sym_penalty_type=sym_penalty_type, sym_layers=sym_layers, lambda_sym=lambda_sym
    )
    
    return all_metric_values, oracle_Q, Q_values


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Set global seed FIRST for full reproducibility
    run_seed = cfg.experiment.seed
    set_global_seed(run_seed)
    
    # Determine output directory
    outdir = cfg.train.get('outdir', None)
    if outdir is not None:
        # Create timestamped subfolder within the specified parent directory
        from datetime import datetime
        import shutil
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        penalty_type = cfg.train.get('sym_penalty_type', 'none')
        lambda_sym = cfg.train.get('lambda_sym', 0.0)
        sym_layers = cfg.train.get('sym_layers', [])
        suffix = cfg.train.get('suffix', None)
        subfolder = f"{timestamp}_{penalty_type}_{lambda_sym}_{sym_layers}"
        if suffix:
            subfolder = f"{subfolder}_{suffix}"
        parent_dir = Path('experiments/train') / outdir
        parent_dir.mkdir(parents=True, exist_ok=True)
        output_dir = parent_dir / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove Hydra's default file handlers and add our own
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                root_logger.removeHandler(handler)
        
        # Add file handler to log to our custom output directory
        file_handler = logging.FileHandler(output_dir / 'train.log')
        file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'))
        root_logger.addHandler(file_handler)
        
        # Remove the empty Hydra output directory
        hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        if hydra_output_dir.exists() and hydra_output_dir != output_dir:
            shutil.rmtree(hydra_output_dir, ignore_errors=True)
    else:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    log.info(OmegaConf.to_yaml(cfg))
    
    # Save config to output directory
    OmegaConf.save(cfg, output_dir / 'config.yaml')
    
    # Save the exact command used to run this script
    command = shlex.join([sys.executable] + sys.argv)
    (output_dir / 'command.txt').write_text(command + '\n')
    
    # Derive separate seeds for different randomness sources
    # (run_seed already set at the start of main)
    data_seed = derive_seed(run_seed, "data")
    model_seed = derive_seed(run_seed, "model")
    augmentation_seed = derive_seed(run_seed, "augmentation")
    
    log.info(f"Run seed: {run_seed} -> data: {data_seed}, model: {model_seed}, augmentation: {augmentation_seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create generators for reproducible randomness
    # Note: These are created AFTER set_global_seed for consistent state
    shuffle_generator = torch.Generator().manual_seed(data_seed)
    augmentation_generator = torch.Generator(device=device).manual_seed(augmentation_seed)
    
    train_loader, val_loader, full_dataset = create_dataloaders(
        n_samples=cfg.data.n_samples,
        r_min=cfg.data.r_min,
        r_max=cfg.data.r_max,
        train_split=cfg.data.train_split,
        batch_size=cfg.train.batch_size,
        seed=data_seed,
        shuffle_generator=shuffle_generator,
    )
    
    # Set model seed right before building for reproducible weight initialization
    set_model_seed(model_seed)
    model = build_model(cfg.model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = create_scheduler(optimizer, cfg.train.total_steps, cfg.train.warmup_steps) if cfg.train.use_scheduler else None
    
    # Setup symmetry penalty if configured
    lambda_sym = cfg.train.get('lambda_sym', 0.0)
    sym_layers = list(cfg.train.get('sym_layers', []))
    sym_penalty_type = cfg.train.get('sym_penalty_type', 'raw')
    n_augmentations = cfg.train.get('n_augmentations_train', 4)
    stopgrad_denominator = cfg.train.get('stopgrad_denominator', True)
    
    symmetry_penalty = None
    if lambda_sym > 0 and sym_layers:
        log.info(f"Using symmetry penalty: type={sym_penalty_type}, lambda={lambda_sym}, layers={sym_layers}")
        symmetry_penalty = create_symmetry_penalty(sym_penalty_type, stopgrad_denominator=stopgrad_denominator)
        
        # For periodic PCA penalty, set reference data for re-fitting
        if isinstance(symmetry_penalty, PeriodicPCAOrbitVariancePenalty):
            X_train = torch.cat([batch[0] for batch in train_loader], dim=0)
            log.info(f"Setting reference data for periodic PCA (refit_interval={symmetry_penalty.refit_interval})")
            symmetry_penalty.set_reference_data(X_train)
    
    # Setup dynamics mode for creating training GIFs
    dynamics_mode = cfg.train.get('dynamics_mode', False)
    dynamics_interval = cfg.train.get('dynamics_interval', 10)
    frame_callback = None
    
    if dynamics_mode:
        dynamics_dir = output_dir / 'dynamics_frames'
        dynamics_dir.mkdir(exist_ok=True)
        log.info(f"Dynamics mode enabled: saving frames every {dynamics_interval} steps to {dynamics_dir}")
        
        # Get data tensors for metric computation
        X_full, y_full = full_dataset.get_numpy()
        X_tensor = torch.tensor(X_full, dtype=torch.float32)
        y_tensor = torch.tensor(y_full, dtype=torch.float32).unsqueeze(1) if y_full.ndim == 1 else torch.tensor(y_full, dtype=torch.float32)
        
        # Pre-compute oracle Q (fast - just variance computation)
        oracle_Q = compute_oracle_Q(X_tensor, y_tensor, full_dataset.scalar_field_fn, n_rotations=32, device=device)
        
        # Get metric configs
        q_cfg = OmegaConf.to_container(cfg.metrics.get("Q", {}), resolve=True)
        total_steps = cfg.train.total_steps
        
        # Get field name for frame titles
        field_name = type(full_dataset.scalar_field_fn).__name__
        
        def make_dynamics_frame(step: int, model_snapshot: nn.Module, history: dict):
            """Generate a summary frame at the current training step."""
            # Compute Q metric with standard errors (MI is too slow - trains a classifier each time)
            q_metric = get_metric("Q", **q_cfg)
            Q_values, Q_stds = q_metric.compute(model_snapshot, X_tensor, device=device, return_std=True)
            
            # Generate frame with fixed x-axis
            plot_run_summary(
                history, Q_values, oracle_Q, model_snapshot, full_dataset, device,
                dynamics_dir / f'frame_{step:06d}.png',
                run_name=f"Step {step}",
                MI_values=None,
                oracle_MI=None,
                xlim=(0, total_steps),
                lambda_sym=lambda_sym,
                field_name=field_name,
                sym_penalty_type=sym_penalty_type,
                sym_layers=sym_layers,
                Q_stds=Q_stds,
            )
            plt.close('all')
        
        frame_callback = make_dynamics_frame
    
    # Gradient alignment tracking (for barrier hypothesis testing)
    grad_align_interval = cfg.train.get('grad_align_interval', 0)
    if grad_align_interval > 0 and lambda_sym > 0 and sym_layers:
        log.info(f"Computing gradient alignment every {grad_align_interval} steps")
    
    history = train_loop(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler, device,
        cfg.train.total_steps, cfg.train.log_interval, cfg.train.eval_interval,
        output_dir, cfg.train.save_best,
        symmetry_penalty=symmetry_penalty,
        lambda_sym=lambda_sym,
        sym_layers=sym_layers,
        n_augmentations=n_augmentations,
        frame_callback=frame_callback,
        frame_interval=dynamics_interval,
        grad_align_interval=grad_align_interval,
        augmentation_generator=augmentation_generator,
    )
    
    log.info(f"\nFinal Train MSE: {history['train_loss'][-1]:.6f}")
    log.info(f"Final Val MSE: {history['val_loss'][-1]:.6f}")
    log.info(f"Final Val MAE: {history['val_mae'][-1]:.4f}")
    if lambda_sym > 0 and sym_layers:
        log.info(f"Final Sym Loss (batch): {history['batch_sym_loss'][-1]:.6f}")
    
    # Create GIF and MP4 from dynamics frames if enabled
    if dynamics_mode:
        sys.path.insert(0, str(Path(__file__).parent))
        from make_gif import create_movie
        gif_path, mp4_path = create_movie(
            input_dir=dynamics_dir,
            pattern="frame_*.png",
            duration=100,
            sort_by="name",
        )
        log.info(f"Created dynamics movie: {gif_path}")
        if mp4_path:
            log.info(f"Created dynamics movie: {mp4_path}")
    
    best_model_path = output_dir / 'model_best.pt'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    
    # Get field name from dataset
    field_name = type(full_dataset.scalar_field_fn).__name__
    
    # Plot loss curves
    sym_penalty_name = type(symmetry_penalty).__name__ if symmetry_penalty else None
    plot_loss_curves(
        history,
        output_dir / 'loss_curves.png',
        sym_penalty_name=sym_penalty_name,
        sym_layers=sym_layers,
        lambda_sym=lambda_sym,
        field_name=field_name,
    )
    
    # Plot regression surface
    plot_regression_surface(
        model, full_dataset, output_dir / 'regression_surface.png', device,
        field_name=field_name,
        sym_penalty_type=sym_penalty_type,
        sym_layers=sym_layers,
        lambda_sym=lambda_sym,
    )
    
    # Compute and plot all metrics
    run_name = cfg.experiment.get('name', None)
    field_name = type(full_dataset.scalar_field_fn).__name__
    all_metrics, oracle_Q, Q_values = compute_and_plot_metrics(
        model, full_dataset, cfg, output_dir, device, run_name=run_name,
        field_name=field_name, sym_penalty_type=sym_penalty_type, sym_layers=sym_layers, lambda_sym=lambda_sym
    )
    
    # Create combined summary plot
    plot_run_summary(
        history, Q_values, oracle_Q, model, full_dataset, device,
        output_dir / 'summary.png', run_name=run_name,
        lambda_sym=lambda_sym,
        field_name=field_name,
        sym_penalty_type=sym_penalty_type,
        sym_layers=sym_layers,
    )
    
    plt.close('all')
    log.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
