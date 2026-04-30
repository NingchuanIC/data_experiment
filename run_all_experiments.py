from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict


MODELS = ["graphsage", "gcn", "appnp", "residualgat"]
DATASETS = {
    "graph_dataset": Path("/home/azureuser/data_experiment/data/build_graph/graph_dataset"),
    "build_graph_news_graph_6_feat": Path("/home/azureuser/data_experiment/data/build_graph_news_graph_6_feat"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 4 models on 2 datasets for multiple runs and summarize mean/variance."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/azureuser/data_experiment"),
        help="Project root that contains train_model.py",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/home/azureuser/data_experiment/Result"),
        help="Output root for all logs and summaries",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per model per dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs passed to train_model.py")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda"], help="Training device")
    return parser.parse_args()


def safe_float(text: str) -> float:
    text = text.strip()
    if text.lower() == "nan":
        return float("nan")
    return float(text)


def parse_key_values(payload: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    parts = [p.strip() for p in payload.split("|")]
    for part in parts:
        if "=" not in part:
            continue
        key, val = [x.strip() for x in part.split("=", 1)]
        try:
            metrics[key] = safe_float(val)
        except ValueError:
            continue
    return metrics


def parse_log_metrics(log_path: Path) -> Dict[str, float]:
    """Parse metrics from one train_model output log."""
    result: Dict[str, float] = {}
    if not log_path.exists():
        return result

    split_line_re = re.compile(r"^\[(train|validation|test)\]\s+(.*)$")
    topk_re = re.compile(r"^Top(\d+)$")

    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("best_val_ic:"):
            _, v = line.split(":", 1)
            try:
                result["best_val_ic"] = safe_float(v)
            except ValueError:
                pass
            continue

        m = split_line_re.match(line)
        if not m:
            continue

        split = m.group(1)
        payload = m.group(2)

        if payload.startswith("samples="):
            try:
                result[f"{split}.samples"] = safe_float(payload.split("=", 1)[1])
            except ValueError:
                pass
            continue

        kv = parse_key_values(payload)
        if "Top" in payload:
            # Example keys: "Top10 IRR", "Top10 Sharpe"
            for k, v in kv.items():
                k = k.replace(" ", "")
                top_match = topk_re.match(k)
                if top_match:
                    continue
                result[f"{split}.{k}"] = v
        else:
            for k, v in kv.items():
                result[f"{split}.{k}"] = v

    return result


def add_value(bucket: Dict[str, list[float]], key: str, val: float) -> None:
    if key not in bucket:
        bucket[key] = []
    bucket[key].append(val)


def summarize_metrics(metric_bucket: Dict[str, list[float]]) -> Dict[str, dict[str, float | int]]:
    summary: Dict[str, dict[str, float | int]] = {}
    for key, values in metric_bucket.items():
        valid = [v for v in values if not math.isnan(v)]
        if not valid:
            summary[key] = {"count": 0, "mean": float("nan"), "variance": float("nan")}
            continue
        if len(valid) == 1:
            var = 0.0
        else:
            var = statistics.pvariance(valid)
        summary[key] = {"count": len(valid), "mean": float(statistics.fmean(valid)), "variance": float(var)}
    return summary


def write_summary_text(path: Path, summary: Dict[str, dict[str, float | int]]) -> None:
    lines = ["Summary (mean/variance over runs)", ""]
    for key in sorted(summary.keys()):
        obj = summary[key]
        lines.append(
            f"{key}: count={obj['count']} mean={obj['mean']:.10f} variance={obj['variance']:.10f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single_experiment(
    project_root: Path,
    model: str,
    dataset_name: str,
    dataset_path: Path,
    run_id: int,
    epochs: int,
    device: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{dataset_name}_run_{run_id:02d}.log"

    cmd = [
        sys.executable,
        "train_model.py",
        "--model",
        model,
        "--graph-data-dir",
        str(dataset_path),
        "--save-model-path",
        "NUL",
        "--output-log",
        str(log_path),
        "--epochs",
        str(epochs),
        "--device",
        device,
    ]

    print(f"[RUN] model={model} dataset={dataset_name} run={run_id:02d}")
    proc = subprocess.run(cmd, cwd=project_root)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Run failed: model={model}, dataset={dataset_name}, run={run_id}, code={proc.returncode}"
        )

    return log_path


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    result_dir = args.result_dir.resolve()

    train_script = project_root / "train_model.py"
    if not train_script.exists():
        raise FileNotFoundError(f"train_model.py not found: {train_script}")

    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    for model in MODELS:
        model_dir = result_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name, dataset_path in DATASETS.items():
            exp_dir = model_dir / dataset_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            metric_bucket: Dict[str, list[float]] = {}
            run_metric_rows: list[dict[str, float]] = []

            for run_id in range(1, args.runs + 1):
                log_path = run_single_experiment(
                    project_root=project_root,
                    model=model,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    run_id=run_id,
                    epochs=args.epochs,
                    device=args.device,
                    output_dir=exp_dir,
                )
                metrics = parse_log_metrics(log_path)
                run_metric_rows.append(metrics)
                for k, v in metrics.items():
                    add_value(metric_bucket, k, v)

            summary = summarize_metrics(metric_bucket)
            (exp_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "dataset": dataset_name,
                        "runs": args.runs,
                        "metrics": summary,
                        "raw_metrics_by_run": run_metric_rows,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            write_summary_text(exp_dir / "summary.txt", summary)
            print(f"[DONE] model={model} dataset={dataset_name} -> {exp_dir}")

    print(f"All experiments finished. Results in: {result_dir}")


if __name__ == "__main__":
    main()
