"""
Simpler script to analyze Ray Tune results and create basic visualizations.
Works directly with Ray Tune's result structure.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_trial_results(experiment_path: str) -> List[Dict]:
    """
    Load all trial results from a Ray Tune experiment directory.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        List of trial result dictionaries
    """
    exp_dir = Path(experiment_path)
    trials = []

    print(f"Scanning experiment directory: {experiment_path}")

    # Iterate through trial directories
    for trial_dir in sorted(exp_dir.iterdir()):
        if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
            continue

        trial_data = {
            'trial_id': trial_dir.name,
            'trial_path': str(trial_dir),
        }

        # Load params.pkl
        params_file = trial_dir / "params.pkl"
        if params_file.exists():
            try:
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
                    # Flatten config params
                    if 'config' in params:
                        for key, value in params['config'].items():
                            trial_data[f'config/{key}'] = value
                    trial_data['params'] = params
            except Exception as e:
                print(f"  Warning: Could not load params from {trial_dir.name}: {e}")

        # Load progress.csv if it exists
        progress_file = trial_dir / "progress.csv"
        if progress_file.exists():
            try:
                progress_df = pd.read_csv(progress_file)
                if len(progress_df) > 0:
                    # Get final metrics
                    final_row = progress_df.iloc[-1]
                    for col in progress_df.columns:
                        trial_data[col] = final_row[col]
            except Exception as e:
                print(f"  Warning: Could not load progress from {trial_dir.name}: {e}")

        # Load result.json if it exists
        result_file = trial_dir / "result.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    # Read last line (final result)
                    lines = f.readlines()
                    if lines:
                        last_result = json.loads(lines[-1])
                        trial_data.update(last_result)
            except Exception as e:
                print(f"  Warning: Could not load result from {trial_dir.name}: {e}")

        if len(trial_data) > 2:  # Has more than just trial_id and trial_path
            trials.append(trial_data)
            print(f"  ✓ Loaded trial: {trial_dir.name}")

    print(f"\nTotal trials loaded: {len(trials)}")
    return trials


def flatten_nested_dict(d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def analyze_results(trials: List[Dict], metric_name: str = "evaluation/custom_metrics/grid2op_end_mean"):
    """
    Analyze trial results and create visualizations.

    Args:
        trials: List of trial dictionaries
        metric_name: Name of the metric to analyze
    """
    # Convert to DataFrame
    df = pd.DataFrame(trials)

    print(f"\n{'='*80}")
    print("TRIAL RESULTS SUMMARY")
    print('='*80)

    print(f"\nTotal trials: {len(df)}")
    print(f"\nAvailable columns: {len(df.columns)}")

    # Find the metric column
    metric_col = None
    for col in df.columns:
        if metric_name in col or col.endswith('grid2op_end_mean'):
            metric_col = col
            break

    if metric_col is None:
        print(f"\nWarning: Could not find metric '{metric_name}'")
        print("\nAvailable metrics:")
        metric_cols = [col for col in df.columns if 'custom_metrics' in col or 'mean' in col]
        for col in metric_cols[:20]:
            print(f"  - {col}")
        if len(metric_cols) > 20:
            print(f"  ... and {len(metric_cols) - 20} more")
        return df

    print(f"\nUsing metric: {metric_col}")

    # Filter trials with valid metric values
    df_valid = df[df[metric_col].notna()].copy()
    print(f"Trials with valid {metric_col}: {len(df_valid)}")

    if len(df_valid) == 0:
        print("No valid trials found!")
        return df

    # Statistics
    print(f"\nMetric Statistics:")
    print(f"  Mean: {df_valid[metric_col].mean():.4f}")
    print(f"  Std:  {df_valid[metric_col].std():.4f}")
    print(f"  Min:  {df_valid[metric_col].min():.4f}")
    print(f"  Max:  {df_valid[metric_col].max():.4f}")

    # Best trial
    best_idx = df_valid[metric_col].idxmax()
    best_trial = df_valid.loc[best_idx]
    print(f"\nBest trial:")
    print(f"  Trial ID: {best_trial['trial_id']}")
    print(f"  {metric_col}: {best_trial[metric_col]:.4f}")

    # Find hyperparameters
    config_cols = [col for col in df_valid.columns if col.startswith('config/model/custom_model_config/gnn/')]
    if config_cols:
        print(f"\n  Hyperparameters:")
        for col in config_cols:
            param_name = col.split('/')[-1]
            print(f"    {param_name}: {best_trial[col]}")

    return df_valid


def create_visualizations(df: pd.DataFrame, metric_col: str, output_dir: str):
    """
    Create visualization plots.

    Args:
        df: DataFrame with trial results
        metric_col: Name of the metric column
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS")
    print('='*80)

    # Set style
    sns.set_style("whitegrid")

    # Find hyperparameter columns
    config_cols = [col for col in df.columns if col.startswith('config/gnn/')]

    if not config_cols:
        print("Warning: No GNN hyperparameter columns found")
        print("Available config columns:")
        config_cols_all = [col for col in df.columns if col.startswith('config/')]
        for col in config_cols_all[:10]:
            print(f"  - {col}")
        config_cols = config_cols_all

    print(f"\nFound {len(config_cols)} hyperparameter columns")

    # 1. Optimization history
    if 'training_iteration' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['training_iteration'], df[metric_col], 'o-', alpha=0.6)
        plt.xlabel('Training Iteration')
        plt.ylabel(metric_col.split('/')[-1])
        plt.title('Optimization History')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'optimization_history.png')
        plt.savefig(output_file, dpi=150)
        print(f"  ✓ Saved: {output_file}")
        plt.close()

    # 2. Distribution of metric values
    plt.figure(figsize=(10, 6))
    plt.hist(df[metric_col].dropna(), bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel(metric_col.split('/')[-1])
    plt.ylabel('Count')
    plt.title('Distribution of Metric Values Across Trials')
    plt.axvline(df[metric_col].mean(), color='red', linestyle='--', label=f'Mean: {df[metric_col].mean():.2f}')
    plt.legend()
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'metric_distribution.png')
    plt.savefig(output_file, dpi=150)
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # 3. Hyperparameter effects
    if config_cols:
        n_params = min(len(config_cols), 6)  # Plot up to 6 parameters
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, col in enumerate(config_cols[:n_params]):
            ax = axes[i]
            param_name = col.split('/')[-1]

            # Group by parameter value and calculate mean metric
            grouped = df.groupby(col)[metric_col].agg(['mean', 'std', 'count'])

            if len(grouped) > 1:
                # Bar plot
                x_vals = grouped.index.astype(str)
                y_vals = grouped['mean']
                yerr = grouped['std']

                ax.bar(x_vals, y_vals, yerr=yerr, alpha=0.7, capsize=5)
                ax.set_xlabel(param_name)
                ax.set_ylabel(metric_col.split('/')[-1])
                ax.set_title(f'Effect of {param_name}')
                ax.grid(True, alpha=0.3, axis='y')

                # Rotate x labels if needed
                if len(x_vals) > 5:
                    ax.tick_params(axis='x', rotation=45)

        # Remove empty subplots
        for i in range(n_params, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        output_file = os.path.join(output_dir, 'hyperparameter_effects.png')
        plt.savefig(output_file, dpi=150)
        print(f"  ✓ Saved: {output_file}")
        plt.close()

    # 4. Correlation heatmap (if we have numeric hyperparameters)
    numeric_cols = config_cols + [metric_col]
    numeric_df = df[numeric_cols].select_dtypes(include=[np.number])

    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(10, 8))
        correlation = numeric_df.corr()

        # Shorten column names for display
        short_names = {col: col.split('/')[-1] for col in correlation.columns}
        correlation = correlation.rename(columns=short_names, index=short_names)

        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1)
        plt.title('Hyperparameter Correlation Matrix')
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(output_file, dpi=150)
        print(f"  ✓ Saved: {output_file}")
        plt.close()

    print(f"\n✓ All visualizations saved to: {output_dir}")


def save_results_csv(df: pd.DataFrame, output_path: str):
    """Save results to CSV file."""
    # Select relevant columns
    cols_to_save = ['trial_id']

    # Add config columns
    config_cols = [col for col in df.columns if col.startswith('config/')]
    cols_to_save.extend(config_cols)

    # Add metric columns
    metric_cols = [col for col in df.columns if 'custom_metrics' in col or 'mean' in col]
    cols_to_save.extend(metric_cols)

    # Filter to existing columns
    cols_to_save = [col for col in cols_to_save if col in df.columns]

    df_export = df[cols_to_save]
    df_export.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Ray Tune hyperparameter optimization results")
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Path to Ray Tune experiment directory"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="evaluation/custom_metrics/grid2op_end_mean",
        help="Metric to analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualization_results",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Path to save results as CSV"
    )

    args = parser.parse_args()

    # Load trials
    trials = load_trial_results(args.experiment_path)

    if not trials:
        print("Error: No trials found!")
        return

    # Analyze results
    df = analyze_results(trials, args.metric)

    if df is not None and len(df) > 0:
        # Find the actual metric column
        metric_col = None
        for col in df.columns:
            if args.metric in col or col.endswith('grid2op_end_mean'):
                metric_col = col
                break

        if metric_col:
            # Create visualizations
            create_visualizations(df, metric_col, args.output_dir)

            # Save CSV
            if args.csv_output:
                save_results_csv(df, args.csv_output)
            else:
                default_csv = os.path.join(args.output_dir, 'results.csv')
                save_results_csv(df, default_csv)


if __name__ == "__main__":
    main()

