import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# =========================
# Data Loading & Utilities
# =========================


def load_and_clean_data(csv_path):
    """Load CSV data and handle missing values."""
    df = pd.read_csv(csv_path)
    df_clean = df.dropna()
    df_clean["Image_Name"] = df_clean["Image"].str.replace(".pgm", "")
    print(f"Loaded {len(df)} rows, {len(df_clean)} valid rows after cleaning")
    return df_clean


def setup_plot_style():
    """Configure matplotlib and seaborn styling."""
    plt.style.use("default")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9


# =========================
# Plotting Functions
# =========================


def plot_timing_by_image(df, output_dir):
    """Plot host/device timing, speedup, and normalized time for each image."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Performance Analysis by Image", fontsize=16, fontweight="bold")

    img_stats = (
        df.groupby("Image_Name")
        .agg({"Host_Time_us": "mean", "Device_Time_us": "mean", "Speedup": "mean"})
        .reset_index()
    )

    # Host vs Device times (line plot)
    axes[0].plot(
        img_stats["Image_Name"],
        img_stats["Host_Time_us"],
        marker="o",
        label="Host",
        color="#FF6B6B",
        linewidth=2,
    )
    axes[0].plot(
        img_stats["Image_Name"],
        img_stats["Device_Time_us"],
        marker="o",
        label="Device",
        color="#4ECDC4",
        linewidth=2,
    )
    axes[0].set_xlabel("Images")
    axes[0].set_ylabel("Average Time (μs)")
    axes[0].set_title("Average Execution Time by Image")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Speedup by image (line plot)
    axes[1].plot(
        img_stats["Image_Name"],
        img_stats["Speedup"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#45B7D1",
        label="Speedup",
    )
    axes[1].set_xlabel("Images")
    axes[1].set_ylabel("Average Speedup")
    axes[1].set_title("Average Speedup by Image")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Normalized time comparison
    host_norm = (img_stats["Host_Time_us"] - img_stats["Host_Time_us"].min()) / (
        img_stats["Host_Time_us"].max() - img_stats["Host_Time_us"].min()
    )
    device_norm = (img_stats["Device_Time_us"] - img_stats["Device_Time_us"].min()) / (
        img_stats["Device_Time_us"].max() - img_stats["Device_Time_us"].min()
    )
    axes[2].plot(
        img_stats["Image_Name"],
        host_norm,
        marker="o",
        label="Host (normalized)",
        color="#FF6B6B",
    )
    axes[2].plot(
        img_stats["Image_Name"],
        device_norm,
        marker="o",
        label="Device (normalized)",
        color="#4ECDC4",
    )
    axes[2].set_xlabel("Images")
    axes[2].set_ylabel("Normalized Time")
    axes[2].set_title("Normalized Execution Time by Image")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "timing_by_image.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_parameter_analysis(df, output_dir):
    """Plot average execution time, speedup, and normalized time for each parameter."""
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
    fig.suptitle("Performance Analysis by Parameters", fontsize=16, fontweight="bold")

    # --- Sigma ---
    sigma_stats = (
        df.groupby("Sigma")
        .agg({"Host_Time_us": "mean", "Device_Time_us": "mean", "Speedup": "mean"})
        .reset_index()
    )
    # Time vs Sigma
    axes[0, 0].plot(
        sigma_stats["Sigma"],
        sigma_stats["Host_Time_us"],
        marker="o",
        label="Host",
        color="#FF6B6B",
    )
    axes[0, 0].plot(
        sigma_stats["Sigma"],
        sigma_stats["Device_Time_us"],
        marker="o",
        label="Device",
        color="#4ECDC4",
    )
    axes[0, 0].set_xlabel("Sigma")
    axes[0, 0].set_ylabel("Avg Time (μs)")
    axes[0, 0].set_title("Execution Time vs Sigma")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    # Speedup vs Sigma
    axes[1, 0].plot(
        sigma_stats["Sigma"],
        sigma_stats["Speedup"],
        marker="o",
        label="Speedup",
        color="#45B7D1",
    )
    axes[1, 0].set_xlabel("Sigma")
    axes[1, 0].set_ylabel("Avg Speedup")
    axes[1, 0].set_title("Speedup vs Sigma")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    # Normalized time vs Sigma
    host_norm = (sigma_stats["Host_Time_us"] - sigma_stats["Host_Time_us"].min()) / (
        sigma_stats["Host_Time_us"].max() - sigma_stats["Host_Time_us"].min()
    )
    device_norm = (
        sigma_stats["Device_Time_us"] - sigma_stats["Device_Time_us"].min()
    ) / (sigma_stats["Device_Time_us"].max() - sigma_stats["Device_Time_us"].min())
    axes[2, 0].plot(
        sigma_stats["Sigma"],
        host_norm,
        marker="o",
        label="Host (normalized)",
        color="#FF6B6B",
    )
    axes[2, 0].plot(
        sigma_stats["Sigma"],
        device_norm,
        marker="o",
        label="Device (normalized)",
        color="#4ECDC4",
    )
    axes[2, 0].set_xlabel("Sigma")
    axes[2, 0].set_ylabel("Normalized Time")
    axes[2, 0].set_title("Normalized Time vs Sigma")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # --- Tmin ---
    tmin_stats = (
        df.groupby("Tmin")
        .agg({"Host_Time_us": "mean", "Device_Time_us": "mean", "Speedup": "mean"})
        .reset_index()
    )
    # Time vs Tmin
    axes[0, 1].plot(
        tmin_stats["Tmin"],
        tmin_stats["Host_Time_us"],
        marker="o",
        label="Host",
        color="#FF6B6B",
    )
    axes[0, 1].plot(
        tmin_stats["Tmin"],
        tmin_stats["Device_Time_us"],
        marker="o",
        label="Device",
        color="#4ECDC4",
    )
    axes[0, 1].set_xlabel("Tmin")
    axes[0, 1].set_ylabel("Avg Time (μs)")
    axes[0, 1].set_title("Execution Time vs Tmin")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    # Speedup vs Tmin
    axes[1, 1].plot(
        tmin_stats["Tmin"],
        tmin_stats["Speedup"],
        marker="o",
        label="Speedup",
        color="#45B7D1",
    )
    axes[1, 1].set_xlabel("Tmin")
    axes[1, 1].set_ylabel("Avg Speedup")
    axes[1, 1].set_title("Speedup vs Tmin")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    # Normalized time vs Tmin
    host_norm = (tmin_stats["Host_Time_us"] - tmin_stats["Host_Time_us"].min()) / (
        tmin_stats["Host_Time_us"].max() - tmin_stats["Host_Time_us"].min()
    )
    device_norm = (
        tmin_stats["Device_Time_us"] - tmin_stats["Device_Time_us"].min()
    ) / (tmin_stats["Device_Time_us"].max() - tmin_stats["Device_Time_us"].min())
    axes[2, 1].plot(
        tmin_stats["Tmin"],
        host_norm,
        marker="o",
        label="Host (normalized)",
        color="#FF6B6B",
    )
    axes[2, 1].plot(
        tmin_stats["Tmin"],
        device_norm,
        marker="o",
        label="Device (normalized)",
        color="#4ECDC4",
    )
    axes[2, 1].set_xlabel("Tmin")
    axes[2, 1].set_ylabel("Normalized Time")
    axes[2, 1].set_title("Normalized Time vs Tmin")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # --- Tmax ---
    tmax_stats = (
        df.groupby("Tmax")
        .agg({"Host_Time_us": "mean", "Device_Time_us": "mean", "Speedup": "mean"})
        .reset_index()
    )
    # Time vs Tmax
    axes[0, 2].plot(
        tmax_stats["Tmax"],
        tmax_stats["Host_Time_us"],
        marker="o",
        label="Host",
        color="#FF6B6B",
    )
    axes[0, 2].plot(
        tmax_stats["Tmax"],
        tmax_stats["Device_Time_us"],
        marker="o",
        label="Device",
        color="#4ECDC4",
    )
    axes[0, 2].set_xlabel("Tmax")
    axes[0, 2].set_ylabel("Avg Time (μs)")
    axes[0, 2].set_title("Execution Time vs Tmax")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    # Speedup vs Tmax
    axes[1, 2].plot(
        tmax_stats["Tmax"],
        tmax_stats["Speedup"],
        marker="o",
        label="Speedup",
        color="#45B7D1",
    )
    axes[1, 2].set_xlabel("Tmax")
    axes[1, 2].set_ylabel("Avg Speedup")
    axes[1, 2].set_title("Speedup vs Tmax")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    # Normalized time vs Tmax
    host_norm = (tmax_stats["Host_Time_us"] - tmax_stats["Host_Time_us"].min()) / (
        tmax_stats["Host_Time_us"].max() - tmax_stats["Host_Time_us"].min()
    )
    device_norm = (
        tmax_stats["Device_Time_us"] - tmax_stats["Device_Time_us"].min()
    ) / (tmax_stats["Device_Time_us"].max() - tmax_stats["Device_Time_us"].min())
    axes[2, 2].plot(
        tmax_stats["Tmax"],
        host_norm,
        marker="o",
        label="Host (normalized)",
        color="#FF6B6B",
    )
    axes[2, 2].plot(
        tmax_stats["Tmax"],
        device_norm,
        marker="o",
        label="Device (normalized)",
        color="#4ECDC4",
    )
    axes[2, 2].set_xlabel("Tmax")
    axes[2, 2].set_ylabel("Normalized Time")
    axes[2, 2].set_title("Normalized Time vs Tmax")
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    # --- Threshold Range ---
    df["Threshold_Range"] = df["Tmax"] - df["Tmin"]
    range_stats = (
        df.groupby("Threshold_Range")
        .agg({"Host_Time_us": "mean", "Device_Time_us": "mean", "Speedup": "mean"})
        .reset_index()
    )
    # Time vs Threshold Range
    axes[0, 3].plot(
        range_stats["Threshold_Range"],
        range_stats["Host_Time_us"],
        marker="o",
        label="Host",
        color="#FF6B6B",
    )
    axes[0, 3].plot(
        range_stats["Threshold_Range"],
        range_stats["Device_Time_us"],
        marker="o",
        label="Device",
        color="#4ECDC4",
    )
    axes[0, 3].set_xlabel("Threshold Range (Tmax - Tmin)")
    axes[0, 3].set_ylabel("Avg Time (μs)")
    axes[0, 3].set_title("Execution Time vs Threshold Range")
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)
    # Speedup vs Threshold Range
    axes[1, 3].plot(
        range_stats["Threshold_Range"],
        range_stats["Speedup"],
        marker="o",
        label="Speedup",
        color="#45B7D1",
    )
    axes[1, 3].set_xlabel("Threshold Range (Tmax - Tmin)")
    axes[1, 3].set_ylabel("Avg Speedup")
    axes[1, 3].set_title("Speedup vs Threshold Range")
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    # Normalized time vs Threshold Range
    host_norm = (range_stats["Host_Time_us"] - range_stats["Host_Time_us"].min()) / (
        range_stats["Host_Time_us"].max() - range_stats["Host_Time_us"].min()
    )
    device_norm = (
        range_stats["Device_Time_us"] - range_stats["Device_Time_us"].min()
    ) / (range_stats["Device_Time_us"].max() - range_stats["Device_Time_us"].min())
    axes[2, 3].plot(
        range_stats["Threshold_Range"],
        host_norm,
        marker="o",
        label="Host (normalized)",
        color="#FF6B6B",
    )
    axes[2, 3].plot(
        range_stats["Threshold_Range"],
        device_norm,
        marker="o",
        label="Device (normalized)",
        color="#4ECDC4",
    )
    axes[2, 3].set_xlabel("Threshold Range (Tmax - Tmin)")
    axes[2, 3].set_ylabel("Normalized Time")
    axes[2, 3].set_title("Normalized Time vs Threshold Range")
    axes[2, 3].legend()
    axes[2, 3].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / "parameter_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_detailed_heatmaps(df, output_dir):
    """Create detailed heatmaps for parameter combinations, showing speedup, device time, and host time."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(36, 16))
    fig.suptitle("Heatmaps by Parameter Combinations", fontsize=20, fontweight="bold")

    # --- SPEEDUP HEATMAPS ---
    # Speedup: Sigma vs Tmin
    speedup_pivot = df.groupby(["Sigma", "Tmin"])["Speedup"].mean().unstack()
    sns.heatmap(
        speedup_pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[0, 0],
        cbar_kws={"label": "Speedup"},
    )
    axes[0, 0].set_title("Speedup: Sigma vs Tmin")
    axes[0, 0].set_ylabel("Sigma")
    axes[0, 0].set_xlabel("Tmin")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Speedup: Sigma vs Tmax
    speedup_pivot2 = df.groupby(["Sigma", "Tmax"])["Speedup"].mean().unstack()
    sns.heatmap(
        speedup_pivot2,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[0, 1],
        cbar_kws={"label": "Speedup"},
    )
    axes[0, 1].set_title("Speedup: Sigma vs Tmax")
    axes[0, 1].set_ylabel("Sigma")
    axes[0, 1].set_xlabel("Tmax")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Speedup: Tmin vs Tmax
    speedup_pivot3 = df.groupby(["Tmin", "Tmax"])["Speedup"].mean().unstack()
    sns.heatmap(
        speedup_pivot3,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[0, 2],
        cbar_kws={"label": "Speedup"},
    )
    axes[0, 2].set_title("Speedup: Tmin vs Tmax")
    axes[0, 2].set_ylabel("Tmin")
    axes[0, 2].set_xlabel("Tmax")
    axes[0, 2].tick_params(axis="x", rotation=45)

    # --- DEVICE TIME HEATMAPS ---
    # Device Time: Sigma vs Tmin
    device_pivot = df.groupby(["Sigma", "Tmin"])["Device_Time_us"].mean().unstack()
    sns.heatmap(
        device_pivot,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=axes[1, 0],
        cbar_kws={"label": "Device Time (μs)"},
    )
    axes[1, 0].set_title("Device Time: Sigma vs Tmin")
    axes[1, 0].set_ylabel("Sigma")
    axes[1, 0].set_xlabel("Tmin")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Device Time: Sigma vs Tmax
    device_pivot2 = df.groupby(["Sigma", "Tmax"])["Device_Time_us"].mean().unstack()
    sns.heatmap(
        device_pivot2,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=axes[1, 1],
        cbar_kws={"label": "Device Time (μs)"},
    )
    axes[1, 1].set_title("Device Time: Sigma vs Tmax")
    axes[1, 1].set_ylabel("Sigma")
    axes[1, 1].set_xlabel("Tmax")
    axes[1, 1].tick_params(axis="x", rotation=45)

    # Device Time: Tmin vs Tmax
    device_pivot3 = df.groupby(["Tmin", "Tmax"])["Device_Time_us"].mean().unstack()
    sns.heatmap(
        device_pivot3,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=axes[1, 2],
        cbar_kws={"label": "Device Time (μs)"},
    )
    axes[1, 2].set_title("Device Time: Tmin vs Tmax")
    axes[1, 2].set_ylabel("Tmin")
    axes[1, 2].set_xlabel("Tmax")
    axes[1, 2].tick_params(axis="x", rotation=45)

    # --- HOST TIME HEATMAPS ---
    # Host Time: Sigma vs Tmin
    host_pivot = df.groupby(["Sigma", "Tmin"])["Host_Time_us"].mean().unstack()
    sns.heatmap(
        host_pivot,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        ax=axes[2, 0],
        cbar_kws={"label": "Host Time (μs)"},
    )
    axes[2, 0].set_title("Host Time: Sigma vs Tmin")
    axes[2, 0].set_ylabel("Sigma")
    axes[2, 0].set_xlabel("Tmin")
    axes[2, 0].tick_params(axis="x", rotation=45)

    # Host Time: Sigma vs Tmax
    host_pivot2 = df.groupby(["Sigma", "Tmax"])["Host_Time_us"].mean().unstack()
    sns.heatmap(
        host_pivot2,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        ax=axes[2, 1],
        cbar_kws={"label": "Host Time (μs)"},
    )
    axes[2, 1].set_title("Host Time: Sigma vs Tmax")
    axes[2, 1].set_ylabel("Sigma")
    axes[2, 1].set_xlabel("Tmax")
    axes[2, 1].tick_params(axis="x", rotation=45)

    # Host Time: Tmin vs Tmax
    host_pivot3 = df.groupby(["Tmin", "Tmax"])["Host_Time_us"].mean().unstack()
    sns.heatmap(
        host_pivot3,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        ax=axes[2, 2],
        cbar_kws={"label": "Host Time (μs)"},
    )
    axes[2, 2].set_title("Host Time: Tmin vs Tmax")
    axes[2, 2].set_ylabel("Tmin")
    axes[2, 2].set_xlabel("Tmax")
    axes[2, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "parameter_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Summary Statistics
# =========================


def generate_summary_stats(df, output_dir):
    """Generate and save summary statistics."""
    stats = {
        "Overall Statistics": {
            "Total Tests": len(df),
            "Mean Speedup": f"{df['Speedup'].mean():.2f}x",
            "Max Speedup": f"{df['Speedup'].max():.2f}x",
            "Min Speedup": f"{df['Speedup'].min():.2f}x",
            "Std Speedup": f"{df['Speedup'].std():.2f}x",
            "Mean Host Time": f"{df['Host_Time_us'].mean():.2f} μs",
            "Mean Device Time": f"{df['Device_Time_us'].mean():.2f} μs",
        }
    }

    # Best and worst performing configurations
    best_config = df.loc[df["Speedup"].idxmax()]
    worst_config = df.loc[df["Speedup"].idxmin()]

    stats["Best Configuration"] = {
        "Image": best_config["Image"],
        "Sigma": best_config["Sigma"],
        "Tmin": best_config["Tmin"],
        "Tmax": best_config["Tmax"],
        "Speedup": f"{best_config['Speedup']:.2f}x",
    }

    stats["Worst Configuration"] = {
        "Image": worst_config["Image"],
        "Sigma": worst_config["Sigma"],
        "Tmin": worst_config["Tmin"],
        "Tmax": worst_config["Tmax"],
        "Speedup": f"{worst_config['Speedup']:.2f}x",
    }

    # Save statistics
    with open(output_dir / "summary_statistics.txt", "w") as f:
        for category, data in stats.items():
            f.write(f"{category}:\n")
            f.write("-" * 40 + "\n")
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

    # Print to console
    print("\n" + "=" * 50)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 50)
    for category, data in stats.items():
        print(f"\n{category}:")
        for key, value in data.items():
            print(f"  {key}: {value}")


# =========================
# Main Entry Point
# =========================


def main():
    # Configuration
    csv_path = Path("results/performance_results.csv")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print("Loading performance data...")
    df = load_and_clean_data(csv_path)

    # Setup plotting style
    setup_plot_style()

    # Generate all plots
    print("Generating timing analysis plots...")
    plot_timing_by_image(df, output_dir)

    print("Generating parameter analysis plots...")
    plot_parameter_analysis(df, output_dir)

    print("Generating detailed heatmaps...")
    plot_detailed_heatmaps(df, output_dir)

    # Generate summary statistics
    print("Generating summary statistics...")
    generate_summary_stats(df, output_dir)

    print(f"\nAnalysis complete! Check the '{output_dir}' directory for:")
    print("  - timing_by_image.png")
    print("  - parameter_analysis.png")
    print("  - parameter_heatmaps.png")
    print("  - summary_statistics.txt")


if __name__ == "__main__":
    main()
