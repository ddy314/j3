from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


class RunMonitor:
    """Incremental monitor; the dashboard never owns the trainer process."""

    def __init__(self, run_dir: str | Path, max_records: int = 50_000) -> None:
        self.run_dir = Path(run_dir)
        if self.run_dir.is_symlink():
            self.run_dir = self.run_dir.resolve()
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.offset = 0
        self.records: list[dict[str, Any]] = []
        self.max_records = max_records

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _refresh_records(self) -> None:
        if not self.metrics_path.exists():
            return
        size = self.metrics_path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.records.clear()
        with self.metrics_path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            for line in handle:
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self.offset = handle.tell()
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]

    def _checkpoints(self) -> list[list[Any]]:
        root = self.run_dir / "checkpoints"
        rows: list[tuple[int, list[Any]]] = []
        if not root.is_dir():
            return []
        for path in root.iterdir():
            if path.name.startswith(".") or path.name == "latest" or not (path / "metadata.json").is_file():
                continue
            try:
                meta = json.loads((path / "metadata.json").read_text())
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(
                (
                    int(meta.get("step", -1)),
                    [
                        path.name,
                        meta.get("step"),
                        meta.get("tokens_seen"),
                        meta.get("reason"),
                        meta.get("created_at"),
                    ],
                )
            )
        return [row for _, row in sorted(rows, reverse=True)]

    @staticmethod
    def _fmt(value: Any, digits: int = 2) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:,.{digits}f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    def _plot(self, axis: str):
        """Build the browser-rendered Plotly figure for the selected x axis.

        ``gr.Plot`` serializes the returned figure with ``Figure.to_json()``. Keeping
        this as a Plotly figure means the browser receives the plot specification and
        can provide hover, zoom, pan, and legend interactions without a raster image.
        """
        import plotly.graph_objects as go

        axis_key, axis_label = {
            "step": ("step", "Step"),
            "tokens": ("tokens_seen", "Tokens seen"),
            "wall time": ("elapsed_time", "Elapsed time (s)"),
        }.get(axis, ("tokens_seen", "Tokens seen"))
        figure = go.Figure()

        for key, label, dash, color in (
            ("train_loss", "Train loss", "solid", "#2563eb"),
            ("smoothed_loss", "Smoothed train loss", "solid", "#0f766e"),
            ("validation_loss", "Validation loss", "dash", "#dc2626"),
        ):
            points = [
                (record.get(axis_key, 0), record[key])
                for record in self.records
                if record.get(key) is not None
            ]
            if not points:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[point[0] for point in points],
                    y=[point[1] for point in points],
                    mode="lines",
                    name=label,
                    line={"color": color, "dash": dash, "width": 2},
                    hovertemplate=f"{axis_label}: %{{x:,.0f}}<br>Loss: %{{y:.4f}}<extra>{label}</extra>",
                )
            )

        figure.update_layout(
            title={
                "text": "Training loss" if self.records else "Waiting for metrics.jsonl",
                "x": 0.02,
                "xanchor": "left",
            },
            template="plotly_white",
            height=420,
            margin={"l": 60, "r": 24, "t": 56, "b": 56},
            hovermode="x unified",
            uirevision=f"loss-{axis}",
            legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        )
        figure.update_xaxes(title_text=axis_label, showgrid=False)
        figure.update_yaxes(title_text="Loss", gridcolor="rgba(148, 163, 184, 0.25)", zeroline=False)
        return figure

    def refresh(self, axis: str = "tokens"):
        self._refresh_records()
        status = self._read_json(self.run_dir / "status.json")
        latest = self.records[-1] if self.records else status.get("current_metrics", {})
        target = int(status.get("tokens_target") or latest.get("tokens_target") or 0)
        seen = int(status.get("tokens_seen") or latest.get("tokens_seen") or 0)
        progress = 100.0 * seen / target if target else 0.0
        overview = (
            f"### {status.get('model_name', 'J3')} / `{status.get('run_id', self.run_dir.name)}`\n\n"
            f"**Status:** `{status.get('status', 'unknown')}`  |  **Device:** `{status.get('device', 'unknown')}`  |  "
            f"**Parameters:** {self._fmt(status.get('parameters'))}\n\n"
            f"**Tokens:** {self._fmt(seen)} / {self._fmt(target)} ({progress:.2f}%)  |  "
            f"**Step:** {self._fmt(status.get('step'))}  |  **Epoch:** {self._fmt(status.get('current_data', {}).get('epoch'))}\n\n"
            f"**Start:** {status.get('started_at') or '—'}  |  "
            f"**Elapsed:** {self._fmt(status.get('elapsed_seconds'), 0)} s  |  "
            f"**Last checkpoint:** {status.get('last_checkpoint_at') or '—'}"
        )
        performance = (
            f"### Performance\n\n"
            f"- Tokens/sec: **{self._fmt(latest.get('tokens_per_sec'))}**\n"
            f"- Step time: **{self._fmt(latest.get('step_time') and latest.get('step_time') * 1000)} ms**\n"
            f"- GPU utilization: **{self._fmt(latest.get('gpu_utilization_pct'))}%**\n"
            f"- VRAM used: **{self._fmt(latest.get('gpu_memory_used_mb'))} MiB** / allocated **{self._fmt(latest.get('gpu_memory_allocated_mb'))} MiB**\n"
            f"- Temperature: **{self._fmt(latest.get('gpu_temperature_c'))} °C**  |  Power: **{self._fmt(latest.get('gpu_power_w'))} W**"
        )
        training = (
            f"### Training\n\n"
            f"- Learning rate: **{self._fmt(latest.get('learning_rate'), 6)}**\n"
            f"- Grad norm: **{self._fmt(latest.get('grad_norm'))}**\n"
            f"- Global batch tokens: **{self._fmt(latest.get('global_batch_tokens'))}**\n"
            f"- Microbatch / accumulation: **{self._fmt(latest.get('micro_batch_size'))} / {self._fmt(latest.get('gradient_accumulation_steps'))}**\n"
            f"- Sequence length: **{self._fmt(latest.get('sequence_length'))}**\n"
            f"- Shard / offset: **{self._fmt(latest.get('current_shard'))} / {self._fmt(latest.get('current_offset'))}**"
        )
        throughput = [record.get("tokens_per_sec") for record in self.records if record.get("tokens_per_sec")]
        recent = throughput[-1] if throughput else None
        latest_elapsed = self.records[-1].get("elapsed_time") if self.records else None

        def window_average(seconds: int) -> float | None:
            if latest_elapsed is None:
                values = throughput[-max(1, len(throughput) // 6) :] if throughput else []
            else:
                values = [
                    float(record["tokens_per_sec"])
                    for record in self.records
                    if record.get("tokens_per_sec")
                    and latest_elapsed - float(record.get("elapsed_time", latest_elapsed)) <= seconds
                ]
            return sum(values) / len(values) if values else None

        eta_seconds = latest.get("eta_seconds")
        eta = "—" if eta_seconds is None else f"{eta_seconds / 3600:.2f} h ({time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() + eta_seconds))})"
        eta_block = (
            f"### ETA\n\n- ETA: **{eta}**\n- Recent throughput: **{self._fmt(recent)} tok/s**\n"
            f"- 1 min avg: **{self._fmt(window_average(60))} tok/s**\n"
            f"- 5 min avg: **{self._fmt(window_average(300))} tok/s**\n"
            f"- 30 min avg: **{self._fmt(window_average(1800))} tok/s**"
        )
        logs_path = self.run_dir / "train.log"
        if logs_path.exists():
            logs = "\n".join(logs_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        else:
            logs = "\n".join(json.dumps(record, ensure_ascii=False) for record in self.records[-20:])
        return overview, self._plot(axis), performance, training, eta_block, self._checkpoints(), logs

    def control(self, action: str) -> str:
        path = self.run_dir / "control.json"
        payload = {"action": action, "requested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        os.replace(temporary, path)
        return f"Requested `{action}` at {payload['requested_at']}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a decoupled Gradio monitor for a J3 run")
    parser.add_argument("--run", default="runs/latest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    if not run_dir.exists() and not run_dir.is_symlink():
        raise SystemExit(f"run directory does not exist: {run_dir}")
    import gradio as gr

    monitor = RunMonitor(run_dir)
    outputs = []
    with gr.Blocks(title="J3 Training Monitor") as demo:
        gr.Markdown("# J3 Training Monitor\nLive, decoupled view of `metrics.jsonl`, `status.json`, and checkpoint metadata.")
        axis = gr.Radio(["tokens", "step", "wall time"], value="tokens", label="Loss X axis")
        overview = gr.Markdown()
        with gr.Row():
            performance = gr.Markdown()
            training = gr.Markdown()
            eta = gr.Markdown()
        loss_plot = gr.Plot(label="Loss")
        checkpoints = gr.Dataframe(
            headers=["checkpoint", "step", "tokens", "reason", "created"],
            datatype=["str", "number", "number", "str", "str"],
            label="Checkpoints",
        )
        with gr.Row():
            save_button = gr.Button("Save checkpoint now")
            stop_button = gr.Button("Request graceful stop", variant="stop")
            control_status = gr.Markdown()
        logs = gr.Textbox(label="Recent logs", lines=14, max_lines=20)
        output_components = [overview, loss_plot, performance, training, eta, checkpoints, logs]
        save_button.click(lambda: monitor.control("checkpoint"), outputs=control_status)
        stop_button.click(lambda: monitor.control("stop"), outputs=control_status)
        axis.change(monitor.refresh, inputs=axis, outputs=output_components)
        timer = gr.Timer(2.0)
        timer.tick(monitor.refresh, inputs=axis, outputs=output_components)
        demo.load(monitor.refresh, inputs=axis, outputs=output_components)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
