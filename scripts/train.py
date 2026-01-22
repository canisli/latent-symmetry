#!/usr/bin/env python3

from latsym.models import MLP
from latsym.tasks import create_dataloaders
from latsym.train import train_loop, create_scheduler, plot_loss_curves
from latsym.eval import plot_regression_surface

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def build_model(cfg: DictConfig) -> nn.Module:
    dims = [cfg.input_dim] + [cfg.hidden_dim] * cfg.num_layers + [cfg.output_dim]
    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
    return MLP(dims=dims, act=act_map.get(cfg.activation, nn.ReLU))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    seed = cfg.experiment.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, full_dataset = create_dataloaders(
        n_samples=cfg.data.n_samples,
        r_min=cfg.data.r_min,
        r_max=cfg.data.r_max,
        train_split=cfg.data.train_split,
        batch_size=cfg.train.batch_size,
        seed=seed,
    )
    
    model = build_model(cfg.model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = create_scheduler(optimizer, cfg.train.total_steps, cfg.train.warmup_steps) if cfg.train.use_scheduler else None
    
    history = train_loop(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler, device,
        cfg.train.total_steps, cfg.train.log_interval, cfg.train.eval_interval,
        output_dir, cfg.train.save_best,
    )
    
    print(f"Final Train MSE: {history['train_loss'][-1]:.6f}")
    print(f"Final Val MSE: {history['val_loss'][-1]:.6f}")
    print(f"Final Val MAE: {history['val_mae'][-1]:.4f}")
    
    best_model_path = output_dir / 'model_best.pt'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    plot_loss_curves(history, output_dir / 'loss_curves.png')
    plot_regression_surface(model, full_dataset, output_dir / 'regression_surface.png', device)

    plt.show()


if __name__ == "__main__":
    main()
