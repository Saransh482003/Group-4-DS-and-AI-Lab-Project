import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERF_LOGS_DIR = os.path.join(BASE_DIR, "perf_logs")
RUN_ID = "20260329_080125_deterministic"  # Set to a specific folder name (e.g., 20260329_080125_deterministic) or keep None for latest.
PLOTS_DIR_NAME = "plots"

MODULE_COLORS = {
    "YOLO": "#0B6E4F",
    "Depth": "#1F78B4",
    "Navigation": "#E67E22",
    "TTS": "#C0392B",
    "App Loop": "#6C5CE7",
    "Frame Total": "#2D3436",
    "Capture": "#16A085",
    "Video Write": "#8E44AD",
    "Display": "#7F8C8D",
    "Visualization": "#D35400",
}

sns.set_theme(
    style="whitegrid",
    context="talk",
    palette="deep",
    rc={
        "figure.facecolor": "#F8FAFC",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D5DBE3",
        "grid.color": "#E6ECF2",
        "grid.alpha": 0.9,
        "axes.titleweight": "semibold",
        "axes.titlepad": 14,
        "axes.labelpad": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    },
)


def _safe_read_jsonl(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records)
    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
    if "timestamp" in df.columns:
        ts0 = df["timestamp"].iloc[0]
        df["elapsed_sec"] = df["timestamp"] - ts0
    return df


def _safe_read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _latest_run_dir(base_dir):
    if not os.path.isdir(base_dir):
        return None
    dirs = [
        os.path.join(base_dir, name)
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]
    if not dirs:
        return None
    dirs.sort(key=os.path.getmtime)
    return dirs[-1]


def _get_run_dir():
    if RUN_ID:
        return os.path.join(PERF_LOGS_DIR, RUN_ID)
    return _latest_run_dir(PERF_LOGS_DIR)


def _save_plot(fig, out_path):
    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _format_axes(ax, title, xlabel, ylabel, subtitle=None):
    ax.set_title(title, fontsize=15, fontweight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.55)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            fontsize=11,
            color="#4B5563",
            ha="left",
            va="bottom",
        )


def _annotate_percentiles(ax, series, color="#334155"):
    if series.empty:
        return
    p50, p95 = series.quantile(0.50), series.quantile(0.95)
    ax.axvline(p50, color=color, linestyle="--", linewidth=1.5, alpha=0.9)
    ax.axvline(p95, color="#B91C1C", linestyle=":", linewidth=1.8, alpha=0.95)
    ylim = ax.get_ylim()
    ax.text(p50, ylim[1] * 0.92, f"p50={p50:.2f}", fontsize=10, color=color)
    ax.text(p95, ylim[1] * 0.85, f"p95={p95:.2f}", fontsize=10, color="#B91C1C")


def _ensure_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def plot_module_latency_distributions(overall_df, yolo_df, depth_df, nav_df, tts_df, out_dir):
    data = []
    sources = [
        ("YOLO", yolo_df, "latency_ms"),
        ("Depth", depth_df, "latency_ms"),
        ("Navigation", nav_df, "latency_ms"),
        ("TTS", tts_df, "latency_ms"),
        ("App Loop", overall_df, "app_loop_latency_ms"),
        ("Frame Total", overall_df, "frame_total_latency_ms"),
        ("Capture", overall_df, "capture_latency_ms"),
        ("Video Write", overall_df, "video_write_latency_ms"),
        ("Display", overall_df, "display_latency_ms"),
        ("Visualization", overall_df, "visualization_latency_ms"),
    ]

    for module, df, col in sources:
        if df.empty or col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        for value in series.tolist():
            data.append({"module": module, "latency_ms": value})

    if not data:
        return

    plot_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(16, 8))
    ordered_modules = [
        "App Loop",
        "Frame Total",
        "YOLO",
        "Depth",
        "Navigation",
        "TTS",
        "Capture",
        "Video Write",
        "Display",
        "Visualization",
    ]
    colors = [MODULE_COLORS.get(name, "#4C78A8") for name in ordered_modules]
    sns.boxplot(
        data=plot_df,
        x="module",
        y="latency_ms",
        order=ordered_modules,
        palette=colors,
        ax=ax,
        showfliers=False,
        linewidth=1.1,
    )
    _format_axes(
        ax,
        "Module-wise Latency Distribution",
        "Module",
        "Latency (ms)",
        subtitle="Outliers hidden to focus on stable operating behavior",
    )
    ax.tick_params(axis="x", rotation=20)
    _save_plot(fig, os.path.join(out_dir, "01_module_latency_boxplot.png"))



def plot_latency_timeline(overall_df, yolo_df, depth_df, nav_df, tts_df, out_dir):
    fig, ax = plt.subplots(figsize=(18, 8))

    if not overall_df.empty and "elapsed_sec" in overall_df.columns:
        if "app_loop_latency_ms" in overall_df.columns:
            sns.lineplot(
                data=overall_df,
                x="elapsed_sec",
                y="app_loop_latency_ms",
                ax=ax,
                label="App Loop",
                alpha=0.55,
                linewidth=1.8,
                color=MODULE_COLORS["App Loop"],
            )
            rolling = overall_df["app_loop_latency_ms"].rolling(30, min_periods=1).mean()
            ax.plot(
                overall_df["elapsed_sec"],
                rolling,
                linewidth=2.8,
                label="App Loop (rolling mean)",
                color="#3B2CBF",
            )

        if "frame_total_latency_ms" in overall_df.columns:
            sns.lineplot(
                data=overall_df,
                x="elapsed_sec",
                y="frame_total_latency_ms",
                ax=ax,
                label="Frame Total",
                alpha=0.35,
                linewidth=1.7,
                color=MODULE_COLORS["Frame Total"],
            )

    if not yolo_df.empty and "elapsed_sec" in yolo_df.columns and "latency_ms" in yolo_df.columns:
        sns.lineplot(
            data=yolo_df,
            x="elapsed_sec",
            y="latency_ms",
            ax=ax,
            label="YOLO",
            alpha=0.85,
            linewidth=1.6,
            color=MODULE_COLORS["YOLO"],
        )
    if not depth_df.empty and "elapsed_sec" in depth_df.columns and "latency_ms" in depth_df.columns:
        sns.lineplot(
            data=depth_df,
            x="elapsed_sec",
            y="latency_ms",
            ax=ax,
            label="Depth",
            alpha=0.85,
            linewidth=1.6,
            color=MODULE_COLORS["Depth"],
        )
    if not nav_df.empty and "elapsed_sec" in nav_df.columns and "latency_ms" in nav_df.columns:
        sns.lineplot(
            data=nav_df,
            x="elapsed_sec",
            y="latency_ms",
            ax=ax,
            label="Navigation",
            alpha=0.85,
            linewidth=1.6,
            color=MODULE_COLORS["Navigation"],
        )
    if not tts_df.empty and "elapsed_sec" in tts_df.columns and "latency_ms" in tts_df.columns:
        sns.lineplot(
            data=tts_df,
            x="elapsed_sec",
            y="latency_ms",
            ax=ax,
            label="TTS",
            alpha=0.85,
            linewidth=1.6,
            color=MODULE_COLORS["TTS"],
        )

    _format_axes(
        ax,
        "Latency Timeline Across Modules",
        "Elapsed Time (sec)",
        "Latency (ms)",
        subtitle="Rolling mean helps surface sustained performance drifts",
    )
    ax.legend(loc="upper right")
    _save_plot(fig, os.path.join(out_dir, "02_latency_timeline.png"))



def plot_overall_histogram(overall_df, out_dir):
    if overall_df.empty or "app_loop_latency_ms" not in overall_df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    series = overall_df["app_loop_latency_ms"].dropna()
    sns.histplot(series, bins=50, kde=True, ax=ax, color=MODULE_COLORS["App Loop"], edgecolor="white")
    _annotate_percentiles(ax, series)
    _format_axes(ax, "App Loop Latency Distribution", "Latency (ms)", "Frame Count")
    _save_plot(fig, os.path.join(out_dir, "03_app_loop_hist_kde.png"))



def plot_fps_timeline(overall_df, out_dir):
    if overall_df.empty or "fps_instant" not in overall_df.columns:
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.lineplot(
        data=overall_df,
        x="elapsed_sec",
        y="fps_instant",
        ax=ax,
        alpha=0.55,
        label="Instant FPS",
        color="#0F766E",
    )
    rolling = overall_df["fps_instant"].rolling(30, min_periods=1).mean()
    ax.plot(overall_df["elapsed_sec"], rolling, linewidth=2.8, label="FPS (rolling mean)", color="#064E3B")
    _format_axes(ax, "FPS Stability Over Time", "Elapsed Time (sec)", "FPS")
    ax.legend(loc="upper right")
    _save_plot(fig, os.path.join(out_dir, "04_fps_timeline.png"))



def plot_resource_usage(resources_df, out_dir):
    if resources_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    if "process_cpu_percent" in resources_df.columns:
        sns.lineplot(data=resources_df, x="elapsed_sec", y="process_cpu_percent", ax=axes[0], color="#7C3AED")
        _format_axes(axes[0], "Process CPU Usage", "Elapsed Time (sec)", "CPU (%)")
    if "process_rss_mb" in resources_df.columns:
        sns.lineplot(data=resources_df, x="elapsed_sec", y="process_rss_mb", ax=axes[1], color="#0EA5E9")
        _format_axes(axes[1], "Process RSS Memory", "Elapsed Time (sec)", "Memory (MB)")
    if "system_memory_percent" in resources_df.columns:
        sns.lineplot(data=resources_df, x="elapsed_sec", y="system_memory_percent", ax=axes[2], color="#D97706")
        _format_axes(axes[2], "System Memory Pressure", "Elapsed Time (sec)", "Memory (%)")
    if "gpu_memory_reserved_mb" in resources_df.columns:
        sns.lineplot(data=resources_df, x="elapsed_sec", y="gpu_memory_reserved_mb", ax=axes[3], color="#DC2626")
        _format_axes(axes[3], "GPU Reserved Memory", "Elapsed Time (sec)", "Memory (MB)")

    fig.suptitle("Resource Utilization Dashboard", fontsize=18, fontweight="semibold", y=1.02)
    _save_plot(fig, os.path.join(out_dir, "05_resource_usage_grid.png"))



def plot_detection_vs_yolo(yolo_df, out_dir):
    if yolo_df.empty:
        return
    required = {"latency_ms", "detection_count"}
    if not required.issubset(set(yolo_df.columns)):
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=yolo_df,
        x="detection_count",
        y="latency_ms",
        ax=ax,
        alpha=0.65,
        color=MODULE_COLORS["YOLO"],
        edgecolor="white",
        s=70,
    )
    sns.regplot(data=yolo_df, x="detection_count", y="latency_ms", ax=ax, scatter=False, color="darkorange")
    _format_axes(
        ax,
        "YOLO Latency vs Detection Count",
        "Detection Count",
        "YOLO Latency (ms)",
        subtitle="Regression line highlights scaling trend with object density",
    )
    _save_plot(fig, os.path.join(out_dir, "06_yolo_latency_vs_detections.png"))



def plot_navigation_commands(nav_df, out_dir):
    if nav_df.empty or "command" not in nav_df.columns:
        return

    counts = nav_df["command"].value_counts().reset_index()
    counts.columns = ["command", "count"]

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(data=counts, x="count", y="command", ax=ax, color="#2563EB")
    _format_axes(ax, "Navigation Command Frequency", "Count", "Command")
    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2.0
        ax.text(width + 0.15, y, f"{int(width)}", va="center", fontsize=10)
    _save_plot(fig, os.path.join(out_dir, "07_navigation_command_counts.png"))



def plot_zone_risk_timeline(nav_df, out_dir):
    if nav_df.empty:
        return

    risk_cols = [
        "zone_risks.left",
        "zone_risks.center",
        "zone_risks.right",
    ]
    if not all(col in nav_df.columns for col in risk_cols):
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(data=nav_df, x="elapsed_sec", y="zone_risks.left", ax=ax, label="Left", color="#2563EB")
    sns.lineplot(data=nav_df, x="elapsed_sec", y="zone_risks.center", ax=ax, label="Center", color="#059669")
    sns.lineplot(data=nav_df, x="elapsed_sec", y="zone_risks.right", ax=ax, label="Right", color="#DC2626")
    _format_axes(
        ax,
        "Zone Risk Progression",
        "Elapsed Time (sec)",
        "Risk Score",
        subtitle="Higher values indicate stronger obstacle pressure in that zone",
    )
    ax.legend(loc="upper right")
    _save_plot(fig, os.path.join(out_dir, "08_zone_risk_timeline.png"))



def plot_tts_behavior(tts_df, out_dir):
    if tts_df.empty:
        return

    if "should_speak" in tts_df.columns:
        tts_df["should_speak_int"] = tts_df["should_speak"].fillna(False).astype(int)
    else:
        tts_df["should_speak_int"] = 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if "latency_ms" in tts_df.columns:
        series = tts_df["latency_ms"].dropna()
        sns.histplot(series, bins=40, kde=True, ax=axes[0], color=MODULE_COLORS["TTS"], edgecolor="white")
        _annotate_percentiles(axes[0], series, color=MODULE_COLORS["TTS"])
        _format_axes(axes[0], "TTS Latency Distribution", "Latency (ms)", "Count")

    sns.scatterplot(data=tts_df, x="elapsed_sec", y="should_speak_int", ax=axes[1], alpha=0.6, color="#475569")
    _format_axes(axes[1], "TTS Trigger Timeline", "Elapsed Time (sec)", "Triggered (1=yes)")

    _save_plot(fig, os.path.join(out_dir, "09_tts_behavior.png"))



def plot_correlation_heatmap(merged_df, out_dir):
    if merged_df.empty:
        return

    numeric_cols = [
        "app_loop_latency_ms",
        "frame_total_latency_ms",
        "capture_latency_ms",
        "video_write_latency_ms",
        "display_latency_ms",
        "fps_instant",
        "yolo_latency_ms",
        "depth_latency_ms",
        "navigation_latency_ms",
        "tts_latency_ms",
        "detection_count",
        "process_cpu_percent",
        "process_rss_mb",
        "system_memory_percent",
        "gpu_memory_reserved_mb",
    ]
    available = [c for c in numeric_cols if c in merged_df.columns]
    if len(available) < 3:
        return

    corr = merged_df[available].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="#E5E7EB",
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    _format_axes(ax, "Correlation Heatmap (Latency, Throughput, Resource)", "", "")
    _save_plot(fig, os.path.join(out_dir, "10_correlation_heatmap.png"))



def plot_percentile_summary(merged_df, out_dir):
    metric_cols = {
        "App Loop": "app_loop_latency_ms",
        "YOLO": "yolo_latency_ms",
        "Depth": "depth_latency_ms",
        "Navigation": "navigation_latency_ms",
        "TTS": "tts_latency_ms",
    }

    rows = []
    for label, col in metric_cols.items():
        if col not in merged_df.columns:
            continue
        series = pd.to_numeric(merged_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append({"module": label, "percentile": "p50", "latency_ms": series.quantile(0.50)})
        rows.append({"module": label, "percentile": "p95", "latency_ms": series.quantile(0.95)})
        rows.append({"module": label, "percentile": "p99", "latency_ms": series.quantile(0.99)})

    if not rows:
        return

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=plot_df, x="module", y="latency_ms", hue="percentile", ax=ax, edgecolor="white")
    _format_axes(
        ax,
        "Latency Percentiles by Module",
        "Module",
        "Latency (ms)",
        subtitle="Use p95/p99 to prioritize tail-latency optimization",
    )
    _save_plot(fig, os.path.join(out_dir, "11_latency_percentiles.png"))



def plot_latency_spike_frames(merged_df, out_dir):
    if merged_df.empty or "app_loop_latency_ms" not in merged_df.columns:
        return

    threshold = merged_df["app_loop_latency_ms"].quantile(0.95)
    spikes = merged_df[merged_df["app_loop_latency_ms"] >= threshold].copy()
    if spikes.empty:
        return

    spikes = spikes.sort_values("app_loop_latency_ms", ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(data=spikes, x="frame", y="app_loop_latency_ms", ax=ax, color="#B91C1C")
    _format_axes(ax, "Top Latency Spike Frames (>= p95)", "Frame", "App Loop Latency (ms)")
    ax.tick_params(axis="x", rotation=45)
    _save_plot(fig, os.path.join(out_dir, "12_latency_spike_frames.png"))


def plot_depth_latency_hist(depth_df, out_dir):
    if depth_df.empty or "latency_ms" not in depth_df.columns:
        return

    series = depth_df["latency_ms"].dropna()
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.histplot(series, bins=45, kde=True, ax=ax, color=MODULE_COLORS["Depth"], edgecolor="white")
    _annotate_percentiles(ax, series, color=MODULE_COLORS["Depth"])
    _format_axes(
        ax,
        "Depth Latency Distribution",
        "Depth Latency (ms)",
        "Frame Count",
        subtitle="Percentile markers make jitter and long tail immediately visible",
    )
    _save_plot(fig, os.path.join(out_dir, "13_depth_latency_hist_kde.png"))


def plot_depth_latency_rolling(depth_df, out_dir):
    if depth_df.empty or "elapsed_sec" not in depth_df.columns or "latency_ms" not in depth_df.columns:
        return

    series = depth_df["latency_ms"].dropna()
    if series.empty:
        return

    rolling = depth_df["latency_ms"].rolling(50, min_periods=1).mean()
    p95 = np.nanpercentile(depth_df["latency_ms"].to_numpy(dtype=float), 95)

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.lineplot(
        data=depth_df,
        x="elapsed_sec",
        y="latency_ms",
        ax=ax,
        color="#93C5FD",
        alpha=0.65,
        linewidth=1.5,
        label="Depth latency",
    )
    ax.plot(depth_df["elapsed_sec"], rolling, color=MODULE_COLORS["Depth"], linewidth=2.8, label="Rolling mean (50f)")
    ax.axhline(p95, color="#DC2626", linestyle="--", linewidth=1.8, label=f"p95={p95:.2f} ms")
    _format_axes(
        ax,
        "Depth Latency Trend and Tail Threshold",
        "Elapsed Time (sec)",
        "Depth Latency (ms)",
        subtitle="Frames above p95 line are likely to contribute to responsiveness degradation",
    )
    ax.legend(loc="upper right")
    _save_plot(fig, os.path.join(out_dir, "14_depth_latency_trend.png"))



def _build_merged_frame(overall_df, yolo_df, depth_df, nav_df, tts_df, resources_df):
    merged = overall_df.copy()

    rename_pairs = [
        (yolo_df, {"latency_ms": "yolo_latency_ms", "detection_count": "detection_count"}),
        (depth_df, {"latency_ms": "depth_latency_ms"}),
        (
            nav_df,
            {
                "latency_ms": "navigation_latency_ms",
                "deterministic_latency_ms": "deterministic_nav_latency_ms",
                "slm_latency_ms": "slm_nav_latency_ms",
            },
        ),
        (tts_df, {"latency_ms": "tts_latency_ms", "should_speak": "tts_should_speak"}),
        (
            resources_df,
            {
                "process_cpu_percent": "process_cpu_percent",
                "process_rss_mb": "process_rss_mb",
                "system_memory_percent": "system_memory_percent",
                "gpu_memory_reserved_mb": "gpu_memory_reserved_mb",
            },
        ),
    ]

    for df, col_map in rename_pairs:
        if df.empty or "frame" not in df.columns:
            continue
        keep = ["frame"] + [c for c in col_map.keys() if c in df.columns]
        if len(keep) <= 1:
            continue
        part = df[keep].rename(columns=col_map)
        merged = merged.merge(part, on="frame", how="left")

    return merged



def write_insights(merged_df, architecture_metrics, out_dir):
    lines = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append("Performance Insights Summary")
    lines.append(f"Generated at: {now}")
    lines.append("")

    frames = int(architecture_metrics.get("frames_processed", 0)) if architecture_metrics else 0
    duration_sec = architecture_metrics.get("duration_sec") if architecture_metrics else None
    avg_fps = architecture_metrics.get("avg_fps") if architecture_metrics else None

    lines.append(f"Frames processed: {frames}")
    lines.append(f"Duration (sec): {duration_sec}")
    lines.append(f"Average FPS: {avg_fps}")

    if not merged_df.empty and "app_loop_latency_ms" in merged_df.columns:
        series = merged_df["app_loop_latency_ms"].dropna()
        if not series.empty:
            lines.append(f"App loop p50 latency (ms): {series.quantile(0.50):.3f}")
            lines.append(f"App loop p95 latency (ms): {series.quantile(0.95):.3f}")
            lines.append(f"App loop p99 latency (ms): {series.quantile(0.99):.3f}")

    module_candidates = {
        "YOLO": "yolo_latency_ms",
        "Depth": "depth_latency_ms",
        "Navigation": "navigation_latency_ms",
        "TTS": "tts_latency_ms",
        "Capture": "capture_latency_ms",
        "Video Write": "video_write_latency_ms",
        "Display": "display_latency_ms",
    }

    module_means = {}
    for label, col in module_candidates.items():
        if col in merged_df.columns:
            s = pd.to_numeric(merged_df[col], errors="coerce").dropna()
            if not s.empty:
                module_means[label] = float(s.mean())

    if module_means:
        lines.append("")
        lines.append("Module mean latencies (ms):")
        for label, value in sorted(module_means.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {label}: {value:.3f}")
        bottleneck = max(module_means.items(), key=lambda x: x[1])
        lines.append(f"Primary latency bottleneck by mean: {bottleneck[0]} ({bottleneck[1]:.3f} ms)")

    if not merged_df.empty and "app_loop_latency_ms" in merged_df.columns:
        p95 = merged_df["app_loop_latency_ms"].quantile(0.95)
        spikes = merged_df[merged_df["app_loop_latency_ms"] >= p95]
        lines.append(f"Latency spike frames (>= p95): {len(spikes)}")
        if "tts_should_speak" in spikes.columns:
            tts_spikes = spikes[spikes["tts_should_speak"] == True]
            lines.append(f"Spike frames with TTS trigger: {len(tts_spikes)}")

    if architecture_metrics:
        peaks = architecture_metrics.get("resource_peaks", {})
        lines.append("")
        lines.append("Peak resource usage:")
        lines.append(f"- Peak process RSS (MB): {peaks.get('peak_process_rss_mb')}")
        lines.append(f"- Peak process CPU (%): {peaks.get('peak_process_cpu_percent')}")
        lines.append(f"- Peak system memory (%): {peaks.get('peak_system_memory_percent')}")
        lines.append(f"- Peak GPU reserved memory (MB): {peaks.get('peak_gpu_memory_reserved_mb')}")

    out_path = os.path.join(out_dir, "insights_summary.txt")
    with open(out_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))



def main():
    run_dir = _get_run_dir()
    if run_dir is None or not os.path.isdir(run_dir):
        raise FileNotFoundError(f"No perf_logs run directory found under: {PERF_LOGS_DIR}")

    plots_dir = os.path.join(run_dir, PLOTS_DIR_NAME)
    os.makedirs(plots_dir, exist_ok=True)

    overall_df = _safe_read_jsonl(os.path.join(run_dir, "overall_metrics.jsonl"))
    yolo_df = _safe_read_jsonl(os.path.join(run_dir, "yolo_metrics.jsonl"))
    depth_df = _safe_read_jsonl(os.path.join(run_dir, "depth_metrics.jsonl"))
    nav_df = _safe_read_jsonl(os.path.join(run_dir, "navigation_metrics.jsonl"))
    tts_df = _safe_read_jsonl(os.path.join(run_dir, "tts_metrics.jsonl"))
    resources_df = _safe_read_jsonl(os.path.join(run_dir, "resources_metrics.jsonl"))
    architecture_metrics = _safe_read_json(os.path.join(run_dir, "architecture_metrics.json"))

    _ensure_numeric(
        overall_df,
        [
            "app_loop_latency_ms",
            "frame_total_latency_ms",
            "capture_latency_ms",
            "video_write_latency_ms",
            "display_latency_ms",
            "fps_instant",
            "visualization_latency_ms",
        ],
    )
    _ensure_numeric(yolo_df, ["latency_ms", "detection_count"])
    _ensure_numeric(depth_df, ["latency_ms"])
    _ensure_numeric(nav_df, ["latency_ms", "deterministic_latency_ms", "slm_latency_ms", "zone_risks.left", "zone_risks.center", "zone_risks.right"])
    _ensure_numeric(tts_df, ["latency_ms"])
    _ensure_numeric(resources_df, ["process_cpu_percent", "process_rss_mb", "system_memory_percent", "gpu_memory_reserved_mb"])

    merged_df = _build_merged_frame(overall_df, yolo_df, depth_df, nav_df, tts_df, resources_df)

    plot_module_latency_distributions(overall_df, yolo_df, depth_df, nav_df, tts_df, plots_dir)
    plot_latency_timeline(overall_df, yolo_df, depth_df, nav_df, tts_df, plots_dir)
    plot_overall_histogram(overall_df, plots_dir)
    plot_fps_timeline(overall_df, plots_dir)
    plot_resource_usage(resources_df, plots_dir)
    plot_detection_vs_yolo(yolo_df, plots_dir)
    plot_navigation_commands(nav_df, plots_dir)
    plot_zone_risk_timeline(nav_df, plots_dir)
    plot_tts_behavior(tts_df, plots_dir)
    plot_correlation_heatmap(merged_df, plots_dir)
    plot_percentile_summary(merged_df, plots_dir)
    plot_latency_spike_frames(merged_df, plots_dir)
    plot_depth_latency_hist(depth_df, plots_dir)
    plot_depth_latency_rolling(depth_df, plots_dir)
    write_insights(merged_df, architecture_metrics, plots_dir)

    print(f"Run directory: {run_dir}")
    print(f"Saved plots and insights to: {plots_dir}")


if __name__ == "__main__":
    main()
