import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_and_clean_data(csv_path):
    """Load CSV data and handle missing values."""
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing values
    df_clean = df.dropna()
    
    # Extract image name without extension
    df_clean['Image_Name'] = df_clean['Image'].str.replace('.pgm', '')
    
    print(f"Loaded {len(df)} rows, {len(df_clean)} valid rows after cleaning")
    return df_clean

def setup_plot_style():
    """Configure matplotlib and seaborn styling."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9

def plot_timing_by_image(df, output_dir):
    """Plot host vs device timing for each image."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Analysis by Image', fontsize=16, fontweight='bold')
    
    # Average times by image
    img_stats = df.groupby('Image_Name').agg({
        'Host_Time_us': 'mean',
        'Device_Time_us': 'mean',
        'Speedup': 'mean'
    }).reset_index()
    
    # Host vs Device times
    x = np.arange(len(img_stats))
    width = 0.35
    
    axes[0,0].bar(x - width/2, img_stats['Host_Time_us'], width, 
                  label='Host', alpha=0.8, color='#FF6B6B')
    axes[0,0].bar(x + width/2, img_stats['Device_Time_us'], width,
                  label='Device', alpha=0.8, color='#4ECDC4')
    axes[0,0].set_xlabel('Images')
    axes[0,0].set_ylabel('Average Time (μs)')
    axes[0,0].set_title('Average Execution Time by Image')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(img_stats['Image_Name'], rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Speedup by image
    axes[0,1].bar(img_stats['Image_Name'], img_stats['Speedup'], 
                  color='#45B7D1', alpha=0.8)
    axes[0,1].set_xlabel('Images')
    axes[0,1].set_ylabel('Average Speedup')
    axes[0,1].set_title('Average Speedup by Image')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Time distribution across all tests
    axes[1,0].hist(df['Host_Time_us'], bins=30, alpha=0.7, 
                   label='Host', color='#FF6B6B', density=True)
    axes[1,0].hist(df['Device_Time_us'], bins=30, alpha=0.7,
                   label='Device', color='#4ECDC4', density=True)
    axes[1,0].set_xlabel('Execution Time (μs)')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Execution Time Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Speedup distribution
    axes[1,1].hist(df['Speedup'], bins=30, alpha=0.8, color='#45B7D1')
    axes[1,1].axvline(df['Speedup'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["Speedup"].mean():.1f}x')
    axes[1,1].set_xlabel('Speedup')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Speedup Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_by_image.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_analysis(df, output_dir):
    """Plot performance metrics vs parameters."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Analysis by Parameters', fontsize=16, fontweight='bold')
    
    # Sigma analysis
    sigma_stats = df.groupby('Sigma').agg({
        'Host_Time_us': 'mean',
        'Device_Time_us': 'mean',
        'Speedup': 'mean'
    }).reset_index()
    
    # Host vs Device by Sigma
    x = np.arange(len(sigma_stats))
    width = 0.35
    axes[0,0].bar(x - width/2, sigma_stats['Host_Time_us'], width,
                  label='Host', alpha=0.8, color='#FF6B6B')
    axes[0,0].bar(x + width/2, sigma_stats['Device_Time_us'], width,
                  label='Device', alpha=0.8, color='#4ECDC4')
    axes[0,0].set_xlabel('Sigma')
    axes[0,0].set_ylabel('Average Time (μs)')
    axes[0,0].set_title('Execution Time vs Sigma')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(sigma_stats['Sigma'])
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Speedup by Sigma
    axes[0,1].plot(sigma_stats['Sigma'], sigma_stats['Speedup'], 
                   marker='o', linewidth=2, markersize=8, color='#45B7D1')
    axes[0,1].set_xlabel('Sigma')
    axes[0,1].set_ylabel('Average Speedup')
    axes[0,1].set_title('Speedup vs Sigma')
    axes[0,1].grid(True, alpha=0.3)
    
    # Tmin analysis
    tmin_stats = df.groupby('Tmin')['Speedup'].mean().reset_index()
    axes[0,2].bar(tmin_stats['Tmin'], tmin_stats['Speedup'], 
                  color='#96CEB4', alpha=0.8)
    axes[0,2].set_xlabel('Tmin')
    axes[0,2].set_ylabel('Average Speedup')
    axes[0,2].set_title('Speedup vs Tmin')
    axes[0,2].grid(True, alpha=0.3)
    
    # Tmax analysis
    tmax_stats = df.groupby('Tmax')['Speedup'].mean().reset_index()
    axes[1,0].bar(tmax_stats['Tmax'], tmax_stats['Speedup'],
                  color='#FFEAA7', alpha=0.8)
    axes[1,0].set_xlabel('Tmax')
    axes[1,0].set_ylabel('Average Speedup')
    axes[1,0].set_title('Speedup vs Tmax')
    axes[1,0].grid(True, alpha=0.3)
    
    # Combined threshold analysis
    df['Threshold_Range'] = df['Tmax'] - df['Tmin']
    threshold_stats = df.groupby('Threshold_Range')['Speedup'].mean().reset_index()
    axes[1,1].bar(threshold_stats['Threshold_Range'], threshold_stats['Speedup'],
                  color='#DDA0DD', alpha=0.8)
    axes[1,1].set_xlabel('Threshold Range (Tmax - Tmin)')
    axes[1,1].set_ylabel('Average Speedup')
    axes[1,1].set_title('Speedup vs Threshold Range')
    axes[1,1].grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr_data = df[['Sigma', 'Tmin', 'Tmax', 'Host_Time_us', 'Device_Time_us', 'Speedup']].corr()
    im = axes[1,2].imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1,2].set_xticks(range(len(corr_data.columns)))
    axes[1,2].set_yticks(range(len(corr_data.columns)))
    axes[1,2].set_xticklabels(corr_data.columns, rotation=45)
    axes[1,2].set_yticklabels(corr_data.columns)
    axes[1,2].set_title('Parameter Correlation Matrix')
    
    # Add correlation values to heatmap
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            axes[1,2].text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                          ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_detailed_heatmaps(df, output_dir):
    """Create detailed heatmaps for parameter combinations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Performance Heatmaps', fontsize=16, fontweight='bold')
    
    # Speedup heatmap by Sigma vs Tmin
    speedup_pivot = df.groupby(['Sigma', 'Tmin'])['Speedup'].mean().unstack()
    sns.heatmap(speedup_pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=axes[0], cbar_kws={'label': 'Speedup'})
    axes[0].set_title('Speedup: Sigma vs Tmin')
    axes[0].set_ylabel('Sigma')
    
    # Speedup heatmap by Sigma vs Tmax
    speedup_pivot2 = df.groupby(['Sigma', 'Tmax'])['Speedup'].mean().unstack()
    sns.heatmap(speedup_pivot2, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=axes[1], cbar_kws={'label': 'Speedup'})
    axes[1].set_title('Speedup: Sigma vs Tmax')
    axes[1].set_ylabel('Sigma')
    
    # Device time heatmap by Tmin vs Tmax
    device_pivot = df.groupby(['Tmin', 'Tmax'])['Device_Time_us'].mean().unstack()
    sns.heatmap(device_pivot, annot=True, fmt='.0f', cmap='Blues',
                ax=axes[2], cbar_kws={'label': 'Device Time (μs)'})
    axes[2].set_title('Device Time: Tmin vs Tmax')
    axes[2].set_ylabel('Tmin')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_stats(df, output_dir):
    """Generate and save summary statistics."""
    stats = {
        'Overall Statistics': {
            'Total Tests': len(df),
            'Mean Speedup': f"{df['Speedup'].mean():.2f}x",
            'Max Speedup': f"{df['Speedup'].max():.2f}x",
            'Min Speedup': f"{df['Speedup'].min():.2f}x",
            'Std Speedup': f"{df['Speedup'].std():.2f}x",
            'Mean Host Time': f"{df['Host_Time_us'].mean():.0f} μs",
            'Mean Device Time': f"{df['Device_Time_us'].mean():.0f} μs"
        }
    }
    
    # Best and worst performing configurations
    best_config = df.loc[df['Speedup'].idxmax()]
    worst_config = df.loc[df['Speedup'].idxmin()]
    
    stats['Best Configuration'] = {
        'Image': best_config['Image'],
        'Sigma': best_config['Sigma'],
        'Tmin': best_config['Tmin'],
        'Tmax': best_config['Tmax'],
        'Speedup': f"{best_config['Speedup']:.2f}x"
    }
    
    stats['Worst Configuration'] = {
        'Image': worst_config['Image'],
        'Sigma': worst_config['Sigma'],
        'Tmin': worst_config['Tmin'],
        'Tmax': worst_config['Tmax'],
        'Speedup': f"{worst_config['Speedup']:.2f}x"
    }
    
    # Save statistics
    with open(output_dir / 'summary_statistics.txt', 'w') as f:
        for category, data in stats.items():
            f.write(f"{category}:\n")
            f.write("-" * 40 + "\n")
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    # Print to console
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*50)
    for category, data in stats.items():
        print(f"\n{category}:")
        for key, value in data.items():
            print(f"  {key}: {value}")

def main():
    # Configuration
    csv_path = Path('results/performance_results.csv')
    output_dir = Path('results/plots')
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