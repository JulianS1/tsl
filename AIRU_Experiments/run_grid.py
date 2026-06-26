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
        'hidden_size': [32, 64],
    },

    # These are the same for every run
    fixed={
        'epochs': 20,
        'patience': 30,
        'window': 12,
        'horizon': 12,
        'batch_size': 64
    },

    skip_errors=True,  # log failures and continue rather than crashing
)


if __name__ == '__main__':
    results = grid.run(results_file='results/grid_results.csv')

    # Print a clean summary table
    df = ExperimentGrid.load_results('results/grid_results.csv')
    ok = df[df['status'] == 'ok']
    if not ok.empty:
        summary_cols = ['dataset', 'model', 'lr', 'hidden_size',
                        'test_mae', 'test_rmse', 'test_mape']
        summary_cols = [c for c in summary_cols if c in ok.columns]
        print("\n" + "=" * 70)
        print("Results summary")
        print("=" * 70)
        print(ok[summary_cols].sort_values('test_mae').to_string(index=False))
