"""Fit Sequential Temperature Scaling (STS) for the DSpark confidence head.

Implements the post-hoc calibration described in the DSpark paper
(Section 3.2.1): for each block position k (left to right), a 1D grid
search selects a temperature scalar that minimizes the Expected
Calibration Error (ECE) of the cumulative prefix-survival probability

    a_k = prod_{i <= k} sigmoid(logit_i / T_i),

keeping the already-fitted temperatures of all preceding positions
fixed. Temperature scaling is order-preserving, so it rectifies the
absolute probability magnitudes without changing the confidence head's
token ranking.

Input records are produced by evaluation runs with
``eval.py --confidence-dump-dir <dir>`` (run them with
``--confidence-threshold 0`` and without ``--confidence-calibration-path``
so the dumped logits are uncalibrated and blocks are untruncated). Fit on
a held-out split, then pass the resulting JSON to
``eval.py --confidence-calibration-path``.

Example:
    python scripts/fit_confidence_calibration.py \
        --records confidence_records/gsm8k/confidence_records.jsonl \
        --output confidence_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


EPS_PROB = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-position confidence temperatures (Sequential "
            "Temperature Scaling) from dumped confidence records."
        ),
    )
    parser.add_argument(
        "--records",
        nargs="+",
        required=True,
        help="One or more confidence_records.jsonl files to fit on.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the fitted calibration.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=20,
        help="Equal-width ECE bins; matches the evaluator's coarse bins.",
    )
    parser.add_argument(
        "--grid-min",
        type=float,
        default=0.05,
        help="Smallest temperature in the search grid.",
    )
    parser.add_argument(
        "--grid-max",
        type=float,
        default=20.0,
        help="Largest temperature in the search grid.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=256,
        help="Number of log-spaced grid points between grid-min and grid-max.",
    )
    return parser.parse_args()


def load_records(paths: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load ragged (logits, labels) rows into padded tensors plus a valid mask."""
    logits_rows: list[list[float]] = []
    label_rows: list[list[int]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                logits = [float(value) for value in record["logits"]]
                labels = [int(value) for value in record["labels"]]
                assert len(logits) == len(labels) and len(logits) > 0, (
                    f"{path}:{line_number} has mismatched or empty "
                    f"logits/labels ({len(logits)} vs {len(labels)})."
                )
                logits_rows.append(logits)
                label_rows.append(labels)
    assert logits_rows, f"No records found in: {', '.join(paths)}"

    block_size = max(len(row) for row in logits_rows)
    num_records = len(logits_rows)
    logits = torch.zeros((num_records, block_size), dtype=torch.float64)
    labels = torch.zeros((num_records, block_size), dtype=torch.float64)
    valid = torch.zeros((num_records, block_size), dtype=torch.bool)
    for row_idx, (logit_row, label_row) in enumerate(zip(logits_rows, label_rows)):
        length = len(logit_row)
        logits[row_idx, :length] = torch.tensor(logit_row, dtype=torch.float64)
        labels[row_idx, :length] = torch.tensor(label_row, dtype=torch.float64)
        valid[row_idx, :length] = True
    return logits, labels, valid


def expected_calibration_error(
    probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_bins: int,
) -> float:
    """Equal-width-bin ECE; mirrors PerPositionConfidenceMetrics."""
    probs = probs.clamp(EPS_PROB, 1.0 - EPS_PROB)
    bin_idx = (probs * num_bins).long().clamp_(0, num_bins - 1)
    ones = torch.ones_like(probs)
    count = torch.zeros(num_bins, dtype=torch.float64).scatter_add_(0, bin_idx, ones)
    pred = torch.zeros(num_bins, dtype=torch.float64).scatter_add_(0, bin_idx, probs)
    target = torch.zeros(num_bins, dtype=torch.float64).scatter_add_(
        0, bin_idx, targets
    )
    total = float(count.sum().item())
    if total <= 0.0:
        return float("nan")
    denom = count.clamp_min(1e-12)
    bin_err = (pred / denom - target / denom).abs()
    return float((bin_err * count).sum().item() / total)


def fit_sequential_temperatures(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    grid: torch.Tensor,
    num_bins: int,
) -> tuple[list[float], list[float], list[float]]:
    block_size = logits.shape[1]
    temperatures: list[float] = []
    ece_before: list[float] = []
    ece_after: list[float] = []
    calibrated_cumprod = torch.ones(logits.shape[0], dtype=torch.float64)
    raw_cumprod = torch.ones(logits.shape[0], dtype=torch.float64)

    for position in range(block_size):
        position_valid = valid[:, position]
        num_valid = int(position_valid.sum().item())
        if num_valid == 0:
            temperatures.append(1.0)
            ece_before.append(float("nan"))
            ece_after.append(float("nan"))
            continue

        position_logits = logits[position_valid, position]
        position_labels = labels[position_valid, position]
        prefix = calibrated_cumprod[position_valid]

        raw_probs = raw_cumprod[position_valid] * torch.sigmoid(position_logits)
        ece_before.append(
            expected_calibration_error(
                raw_probs,
                position_labels,
                num_bins=num_bins,
            )
        )

        best_temperature = 1.0
        best_ece = float("inf")
        for temperature in grid.tolist():
            probs = prefix * torch.sigmoid(position_logits / temperature)
            ece = expected_calibration_error(
                probs,
                position_labels,
                num_bins=num_bins,
            )
            if ece < best_ece:
                best_ece = ece
                best_temperature = float(temperature)
        temperatures.append(best_temperature)
        ece_after.append(best_ece)

        calibrated_step = torch.sigmoid(logits[:, position] / best_temperature)
        raw_step = torch.sigmoid(logits[:, position])
        calibrated_cumprod = torch.where(
            position_valid,
            calibrated_cumprod * calibrated_step,
            calibrated_cumprod,
        )
        raw_cumprod = torch.where(
            position_valid,
            raw_cumprod * raw_step,
            raw_cumprod,
        )
        print(
            f"position {position}: n={num_valid} T={best_temperature:.4f} "
            f"cumulative ECE {ece_before[-1]:.4f} -> {best_ece:.4f}",
            flush=True,
        )
    return temperatures, ece_before, ece_after


def main() -> None:
    args = parse_args()
    logits, labels, valid = load_records(args.records)

    grid = torch.logspace(
        math.log10(args.grid_min),
        math.log10(args.grid_max),
        steps=int(args.grid_size),
        dtype=torch.float64,
    )
    # Always include the identity temperature so calibration can no-op.
    grid = torch.unique(torch.cat([grid, torch.ones(1, dtype=torch.float64)]))

    temperatures, ece_before, ece_after = fit_sequential_temperatures(
        logits=logits,
        labels=labels,
        valid=valid,
        grid=grid,
        num_bins=int(args.num_bins),
    )

    payload = {
        "temperatures": temperatures,
        "block_size": logits.shape[1],
        "num_bins": int(args.num_bins),
        "grid": {
            "min": float(args.grid_min),
            "max": float(args.grid_max),
            "size": int(args.grid_size),
        },
        "num_records": logits.shape[0],
        "sources": [str(path) for path in args.records],
        "cumulative_ece_before": ece_before,
        "cumulative_ece_after": ece_after,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Wrote calibration to {output_path}", flush=True)


if __name__ == "__main__":
    main()
