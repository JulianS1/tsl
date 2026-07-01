import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AIRU_Experiments.grid import ExperimentGrid


grid = ExperimentGrid(
    datasets=['la'],
    models=['gru_diffconv'],

    # Every combination of these values will be run
    hparams={
        'lr': [0.001],
        'hidden_size': [64],
        'scaling_mode': ['fixed', 'uniform', 'linear', 'mlp', 'tcn'],
    },

    # These are the same for every run
    fixed={
        'epochs': 20,
        'patience': 30,
        'window': 12,
        'horizon': 12,
        'batch_size': 64
    },

    seeds=[42, 123, 456],

    skip_errors=True,  # log failures and continue rather than crashing
)


if __name__ == '__main__':
    results = grid.run(results_file='results/grid_results.csv')

    #To print mean ± std across seeds
    summary = ExperimentGrid.summarize('results/grid_results.csv')
    summary_cols = ['dataset', 'model', 'scaling_mode', 'hidden_size',
                    'test_mae', 'test_mae_at_15', 'test_mae_at_30', 'test_mae_at_60']
    summary_cols = [c for c in summary_cols if c in summary.columns]
    if not summary.empty:
        print("\n" + "=" * 70)
        print("Results summary (mean ± std across seeds)")
        print("=" * 70)
        print(summary[summary_cols].to_string(index=False))
