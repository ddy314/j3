from __future__ import annotations

import json
from pathlib import Path

from dashboard import RunMonitor


def test_loss_plot_serializes_to_interactive_plotly_spec(tmp_path: Path) -> None:
    monitor = RunMonitor(tmp_path)
    monitor.records = [
        {
            "step": 1,
            "tokens_seen": 128,
            "elapsed_time": 0.5,
            "train_loss": 4.2,
            "smoothed_loss": 4.1,
            "validation_loss": None,
        },
        {
            "step": 2,
            "tokens_seen": 256,
            "elapsed_time": 1.0,
            "train_loss": 3.8,
            "smoothed_loss": 4.0,
            "validation_loss": 3.9,
        },
    ]

    specification = json.loads(monitor._plot("tokens").to_json())

    assert [trace["name"] for trace in specification["data"]] == [
        "Train loss",
        "Smoothed train loss",
        "Validation loss",
    ]
    assert specification["data"][0]["x"] == [128, 256]
    assert specification["data"][0]["y"] == [4.2, 3.8]
    assert specification["layout"]["xaxis"]["title"]["text"] == "Tokens seen"
    assert specification["layout"]["uirevision"] == "loss-tokens"
