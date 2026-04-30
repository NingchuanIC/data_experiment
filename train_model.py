from __future__ import annotations

import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models import get_model


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[0]
    parser = argparse.ArgumentParser(description="Centralized trainer that selects model by name.")
    parser.add_argument("--model", type=str, default="graphsage", help="Model name: graphsage|gcn|appnp|residualgat")
    parser.add_argument("--graph-data-dir", type=Path, default=repo_root / "build_graph" / "graph_dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--K", type=int, default=5, help="K for APPNP")
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for APPNP")
    parser.add_argument("--mse-weight", type=float, default=0.2)
    parser.add_argument("--ic-weight", type=float, default=0.8)
    parser.add_argument("--loss-fn", type=str, choices=["mse", "ic", "mixed_ic_mse"], default="mixed_ic_mse")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--topk-list", type=str, default="10,30,50")
    parser.add_argument("--device", type=str, choices=["cuda"], default="cuda")
    parser.add_argument("--save-model-path", type=Path, default=repo_root / "model.pt")
    parser.add_argument("--output-log", type=Path, default=repo_root / "train_log.txt")
    return parser.parse_args()


def parse_topk_list(topk_list: str) -> list[int]:
    values: list[int] = []
    for token in topk_list.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k <= 0:
            raise ValueError("All Top-k values must be positive integers.")
        values.append(k)

    if not values:
        raise ValueError("--topk-list must contain at least one positive integer.")

    return list(dict.fromkeys(values))


def resolve_device(device_name: str) -> torch.device:
    if device_name != "cuda":
        raise ValueError("This script requires --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) <= 1:
        return float("nan")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mixed_ic_mse_loss(pred: torch.Tensor, target: torch.Tensor, mse_weight: float = 0.2, ic_weight: float = 0.8) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    pred_cent = pred - pred.mean()
    target_cent = target - target.mean()
    pred_std = pred_cent.std(unbiased=False) + 1e-8
    target_std = target_cent.std(unbiased=False) + 1e-8
    ic_loss = -torch.mean(pred_cent * target_cent) / (pred_std * target_std)
    return mse_weight * mse + ic_weight * ic_loss


def negative_corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    pred_std = pred.std(unbiased=False) + 1e-8
    target_std = target.std(unbiased=False) + 1e-8
    corr = torch.mean(pred * target) / (pred_std * target_std)
    return -corr


def get_loss_value(loss_fn: str, pred: torch.Tensor, target: torch.Tensor, mse_weight: float, ic_weight: float) -> torch.Tensor:
    if loss_fn == "mse":
        return F.mse_loss(pred, target)
    if loss_fn == "ic":
        return negative_corr_loss(pred, target)
    if loss_fn == "mixed_ic_mse":
        return mixed_ic_mse_loss(pred, target, mse_weight=mse_weight, ic_weight=ic_weight)
    raise ValueError(f"Unknown loss function: {loss_fn}")


def load_graph_dataset(graph_data_dir: Path):
    if not graph_data_dir.exists():
        raise FileNotFoundError(f"Graph dataset directory not found: {graph_data_dir}")

    meta_path = graph_data_dir / "metadata.pt"
    train_path = graph_data_dir / "train.pt"
    val_path = graph_data_dir / "val.pt"
    test_path = graph_data_dir / "test.pt"

    for p in [meta_path, train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required dataset file not found: {p}")

    metadata = torch.load(meta_path, map_location="cpu")
    train_set = torch.load(train_path, map_location="cpu")
    val_set = torch.load(val_path, map_location="cpu")
    test_set = torch.load(test_path, map_location="cpu")

    return metadata, train_set, val_set, test_set


def predict_split(model: nn.Module, split_set: dict, edge_index: torch.Tensor, device: torch.device):
    x_all = split_set["x"]
    y_all = split_set["y"]
    mask_all = split_set["mask"]
    trade_dates = split_set["trade_dates"]

    edge_index = edge_index.to(device)
    model.eval()
    all_true: list[float] = []
    all_pred: list[float] = []
    all_dates: list[str] = []

    with torch.no_grad():
        for t in range(x_all.shape[0]):
            x = x_all[t].to(device)
            y = y_all[t].to(device)
            mask = mask_all[t].to(device)

            pred = model(x, edge_index).squeeze(-1)
            if int(mask.sum().item()) == 0:
                continue

            y_np = y[mask].cpu().numpy().astype(np.float64)
            pred_np = pred[mask].cpu().numpy().astype(np.float64)
            all_true.extend(y_np.tolist())
            all_pred.extend(pred_np.tolist())
            all_dates.extend([str(trade_dates[t])] * len(y_np))

    return np.asarray(all_true, dtype=np.float64), np.asarray(all_pred, dtype=np.float64), pd.Series(all_dates)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_true - y_pred
    mse = float(np.mean(residual**2))
    sign_accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    ic = safe_corr(y_pred, y_true)
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=np.float64)
    true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float64)
    rankic = safe_corr(pred_rank, true_rank)

    strategy_ret = np.sign(y_pred) * y_true
    ret_std = float(np.std(strategy_ret, ddof=1)) if len(strategy_ret) > 1 else 0.0
    sharpe = float(np.mean(strategy_ret) / ret_std * np.sqrt(252.0)) if ret_std > 0 else float("nan")

    return {"mse": mse, "ic": ic, "rankic": rankic, "sign_accuracy": sign_accuracy, "sharpe": sharpe}


def topk_backtest_metrics(y_true: np.ndarray, y_pred: np.ndarray, trade_date: pd.Series, topk: int) -> dict:
    df_bt = pd.DataFrame({"trade_date": trade_date.to_numpy(), "y_true": y_true, "y_pred": y_pred})

    daily_ret: list[float] = []
    for _, group in df_bt.groupby("trade_date", sort=True):
        picked = group.nlargest(topk, "y_pred")
        if picked.empty:
            continue
        daily_ret.append(float(picked["y_true"].mean()))

    if not daily_ret:
        return {"irr": float("nan"), "sharpe": float("nan")}

    daily_ret_arr = np.asarray(daily_ret, dtype=np.float64)
    gross = 1.0 + daily_ret_arr
    if np.any(gross <= 0):
        irr = float("nan")
    else:
        irr = float(np.prod(gross) ** (252.0 / len(daily_ret_arr)) - 1.0)

    ret_std = float(np.std(daily_ret_arr, ddof=1)) if len(daily_ret_arr) > 1 else 0.0
    sharpe = float(np.mean(daily_ret_arr) / ret_std * np.sqrt(252.0)) if ret_std > 0 else float("nan")

    return {"irr": irr, "sharpe": sharpe}


def train(
    model: nn.Module,
    train_set: dict,
    val_set: dict,
    edge_index: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    early_stopping_patience: int,
    mse_weight: float,
    ic_weight: float,
    loss_fn: str,
):
    model.to(device)
    edge_index = edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_all = train_set["x"]
    y_all = train_set["y"]
    mask_all = train_set["mask"]

    n_steps = x_all.shape[0]

    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_ic = float("-inf")
    best_epoch = 0
    patience = 0

    val_y_true, val_y_pred, _ = predict_split(model, val_set, edge_index, device)
    if len(val_y_true) == 0:
        raise ValueError("validation split is empty. Cannot perform early stopping.")

    logs: list[str] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_point_cnt = 0

        for t in range(n_steps):
            x = x_all[t].to(device)
            y = y_all[t].to(device)
            mask = mask_all[t].to(device)

            pred = model(x, edge_index).squeeze(-1)
            if int(mask.sum().item()) == 0:
                continue

            loss = get_loss_value(loss_fn=loss_fn, pred=pred[mask], target=y[mask], mse_weight=mse_weight, ic_weight=ic_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_obs = int(mask.sum().item())
            epoch_loss_sum += float(loss.item()) * n_obs
            epoch_point_cnt += n_obs

        avg_loss = epoch_loss_sum / max(epoch_point_cnt, 1)
        val_y_true, val_y_pred, _ = predict_split(model, val_set, edge_index, device)
        val_metrics = regression_metrics(val_y_true, val_y_pred)
        current_val_ic = val_metrics["ic"]

        if current_val_ic > best_val_ic:
            best_val_ic = current_val_ic
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        line = f"epoch={epoch:03d} train_loss={avg_loss:.8f} | val_ic={current_val_ic:.8f} | best_val_ic={best_val_ic:.8f} | best_epoch={best_epoch:03d}"
        logs.append(line)
        print(line)

        if patience >= early_stopping_patience:
            logs.append(f"early_stopping_triggered_at_epoch={epoch:03d}")
            print(f"early_stopping_triggered_at_epoch={epoch:03d}")
            break

    return logs, best_state_dict, best_val_ic, best_epoch


def evaluate_split(name: str, model: nn.Module, split_set: dict, edge_index: torch.Tensor, device: torch.device, topk_values: list[int]) -> list[str]:
    y_true, y_pred, trade_date = predict_split(model, split_set, edge_index, device)
    if len(y_true) == 0:
        return [f"[{name}] samples=0", f"[{name}] no valid labeled samples for evaluation"]

    metrics = regression_metrics(y_true, y_pred)
    lines = [f"[{name}] samples={len(y_true)}", f"[{name}] IC={metrics['ic']:.8f} | RankIC={metrics['rankic']:.8f} | SignAcc={metrics['sign_accuracy']:.8f} | Sharpe={metrics['sharpe']:.8f} | MSE={metrics['mse']:.8f}"]

    for k in topk_values:
        bt_metrics = topk_backtest_metrics(y_true, y_pred, trade_date, k)
        lines.append(f"[{name}] Top{k} IRR={bt_metrics['irr']:.8f} | Top{k} Sharpe={bt_metrics['sharpe']:.8f}")

    return lines


def save_outputs(save_model_path: Path, output_log: Path, model: nn.Module, feature_cols: list[str], target_col: str, node_codes: list[str], edge_index: torch.Tensor, result_lines: list[str], model_name: str) -> None:
    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "target_col": target_col,
        "node_codes": node_codes,
        "edge_index": edge_index,
        "model": model_name,
    }
    torch.save(ckpt, save_model_path)

    with output_log.open("w", encoding="utf-8") as f:
        for line in result_lines:
            f.write(line + "\n")


def main() -> None:
    args = parse_args()
    topk_values = parse_topk_list(args.topk_list)
    device = resolve_device(args.device)

    metadata, train_set, val_set, test_set = load_graph_dataset(args.graph_data_dir)

    edge_index = metadata["edge_index"]
    node_codes = metadata["node_codes"]
    feature_cols = metadata["feature_cols"]
    target_col = metadata["target_col"]

    model = get_model(
        name=args.model,
        in_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        K=args.K,
        alpha=args.alpha,
    )

    train_logs, best_state_dict, best_val_ic, best_epoch = train(
        model=model,
        train_set=train_set,
        val_set=val_set,
        edge_index=edge_index,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        mse_weight=args.mse_weight,
        ic_weight=args.ic_weight,
        loss_fn=args.loss_fn,
    )
    model.load_state_dict(best_state_dict)

    result_lines = [
        f"Model: {args.model}",
        f"target: {target_col}",
        f"epochs: {args.epochs}",
        f"lr: {args.lr}",
        f"weight_decay: {args.weight_decay}",
        f"hidden_dim: {args.hidden_dim}",
        f"loss_fn: {args.loss_fn}",
        f"mse_weight: {args.mse_weight}",
        f"ic_weight: {args.ic_weight}",
        f"early_stopping_patience: {args.early_stopping_patience}",
        f"topk_list: {topk_values}",
        f"features ({len(feature_cols)}): {feature_cols}",
        f"best_epoch: {best_epoch}",
        f"best_val_ic: {best_val_ic:.8f}",
    ]
    result_lines.extend(train_logs)
    result_lines.extend(evaluate_split("train", model, train_set, edge_index, device, topk_values))
    result_lines.extend(evaluate_split("validation", model, val_set, edge_index, device, topk_values))
    result_lines.extend(evaluate_split("test", model, test_set, edge_index, device, topk_values))

    save_outputs(
        save_model_path=args.save_model_path,
        output_log=args.output_log,
        model=model,
        feature_cols=feature_cols,
        target_col=target_col,
        node_codes=node_codes,
        edge_index=edge_index,
        result_lines=result_lines,
        model_name=args.model,
    )

    print("Training finished.")


if __name__ == "__main__":
    main()
