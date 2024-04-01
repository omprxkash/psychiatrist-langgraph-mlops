"""
PyTorch tabular MLP for severity classification + auxiliary ordinal regression.

Architecture:
  - Shared encoder: Linear → BN → GELU → Dropout (×3 layers)
  - Classification head: softmax over severity bands
  - Aux regression head: single scalar (phq9_total) for multi-task learning
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class MLPConfig:
    input_dim: int
    num_classes: int
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_size: int = 512
    aux_weight: float = 0.2
    patience: int = 8
    device: str = "cpu"
    seed: int = 42


class SeverityMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        dims = [cfg.input_dim] + cfg.hidden_dims
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:], strict=False):
            layers += [
                nn.Linear(in_d, out_d),
                nn.BatchNorm1d(out_d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
        self.encoder = nn.Sequential(*layers)
        self.clf_head = nn.Linear(cfg.hidden_dims[-1], cfg.num_classes)
        self.reg_head = nn.Linear(cfg.hidden_dims[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.clf_head(h), self.reg_head(h).squeeze(-1)


def _make_loader(x, y, y_ord, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(y_ord, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_ord_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_ord_val: np.ndarray,
    cfg: MLPConfig,
    run_name: str = "torch-mlp-severity",
) -> tuple[SeverityMLP, dict]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = SeverityMLP(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    clf_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    train_loader = _make_loader(x_train, y_train, y_ord_train, cfg.batch_size, shuffle=True)
    val_loader = _make_loader(x_val, y_val, y_ord_val, cfg.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    history: list[dict] = []

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(
            {
                "hidden_dims": str(cfg.hidden_dims),
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "aux_weight": cfg.aux_weight,
            }
        )

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, yb, yordb in train_loader:
                xb, yb, yordb = xb.to(device), yb.to(device), yordb.to(device)
                optimizer.zero_grad()
                logits, reg_out = model(xb)
                loss = clf_loss_fn(logits, yb) + cfg.aux_weight * reg_loss_fn(reg_out, yordb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
            scheduler.step()

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb, yordb in val_loader:
                    xb, yb, yordb = xb.to(device), yb.to(device), yordb.to(device)
                    logits, reg_out = model(xb)
                    loss = clf_loss_fn(logits, yb) + cfg.aux_weight * reg_loss_fn(reg_out, yordb)
                    val_loss += loss.item() * len(xb)
                    correct += (logits.argmax(1) == yb).sum().item()
                    total += len(xb)

            t_loss = train_loss / len(train_loader.dataset)
            v_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total

            mlflow.log_metrics(
                {"train_loss": t_loss, "val_loss": v_loss, "val_acc": val_acc}, step=epoch
            )
            history.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss})

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        metrics = _final_metrics(model, val_loader, device)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, artifact_path="torch-mlp-severity")

    return model, metrics


def _final_metrics(model: SeverityMLP, loader: DataLoader, device) -> dict:
    from sklearn.metrics import cohen_kappa_score, f1_score

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            logits, _ = model(xb.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_targets.extend(yb.tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    return {
        "val_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "val_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "val_quadratic_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def predict_proba(model: SeverityMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor(x, dtype=torch.float32))
        return torch.softmax(logits, dim=-1).numpy()
