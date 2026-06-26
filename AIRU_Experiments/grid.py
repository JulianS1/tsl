import csv
import itertools
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Defaults mirror parse_args() and match the GWN paper baseline (Wu et al., IJCAI 2019).
DEFAULTS = {
    'connectivity_layout': 'edge_index',
    'window': 12,
    'horizon': 12,
    'stride': 1,
    'epochs': 100,
    'batch_size': 64,
    'lr': 0.001,
    'lr_decay': 0.97,
    'weight_decay': 0.0001,
    'grad_clip': 5.0,
    'patience': 50,
    'workers': 0,
    'seed': 42,
    'log_dir': 'logs/experiment',
    'logger_backend': None,
    'accelerator': 'auto',
    'results_file': None,
    'hidden_size': 32,
    'ff_size': 256,
    'n_layers': 8,
    'emb_size': 10,
    'dropout': 0.3,
    'temporal_kernel_size': 2,
    'spatial_kernel_size': 2,
    'dilation': 2,
    'dilation_mod': 2,
    'norm': 'batch',
}


def _append_to_csv(row: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class ExperimentGrid:
    """Define and run a grid of experiments.

    Example usage::

        grid = ExperimentGrid(
            datasets=['la', 'bay'],
            models=['gwnet', 'fcrnn'],
            hparams={
                'lr': [0.001, 0.0001],
                'hidden_size': [32, 64],
            },
            fixed={'epochs': 100, 'patience': 20},
        )
        results = grid.run('results/my_grid.csv')

    Args:
        datasets: List of dataset names to sweep over.
        models: List of model names to sweep over.
        hparams: Dict mapping hyperparameter names to lists of values.
            Every combination of values will be tried.
        fixed: Dict of hyperparameters that are the same for every run.
            These override the defaults but are not swept.
        skip_errors: If True, log failed runs and continue. If False, raise.
    """

    def __init__(self, datasets, models, hparams=None, fixed=None,
                 skip_errors=True):
        self.datasets = datasets
        self.models = models
        self.hparams = hparams or {}
        self.fixed = fixed or {}
        self.skip_errors = skip_errors

    def _configs(self):
        """Yield one config dict per experiment."""
        hparam_names = list(self.hparams.keys())
        hparam_values = list(self.hparams.values())
        combos = list(itertools.product(*hparam_values)) if hparam_values else [()]

        for dataset in self.datasets:
            for model in self.models:
                for combo in combos:
                    cfg = dict(DEFAULTS)
                    cfg.update(self.fixed)
                    cfg['dataset'] = dataset
                    cfg['model'] = model
                    for name, value in zip(hparam_names, combo):
                        cfg[name] = value
                    yield cfg

    def run(self, results_file='results/grid_results.csv'):
        """Run all experiments and save results to a CSV.

        Results are appended after each run so partial results survive crashes.

        Args:
            results_file: Path to the output CSV file.

        Returns:
            List of result dicts (one per completed experiment).
        """
        from AIRU_Experiments.main import run_experiment

        configs = list(self._configs())
        total = len(configs)
        all_results = []

        print(f"Running {total} experiments → {results_file}")

        for i, cfg in enumerate(configs, 1):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            cfg['log_dir'] = f"logs/{cfg['model']}/{cfg['dataset']}/{timestamp}"

            print(f"\n[{i}/{total}] {cfg['model']} | {cfg['dataset']} | "
                  + " | ".join(f"{k}={v}" for k, v in self.hparams.items()
                               if k in cfg))

            args = Namespace(**cfg)
            try:
                res = run_experiment(args)
                row = {**cfg, **res, 'status': 'ok', 'timestamp': timestamp}
                print(f"  → test_mae={res['test_mae']:.4f}  "
                      f"test_rmse={res['test_rmse']:.4f}  "
                      f"test_mape={res['test_mape']:.4f}")
            except Exception as e:
                print(f"  FAILED: {e}")
                if not self.skip_errors:
                    raise
                row = {**cfg, 'status': f'error: {e}', 'timestamp': timestamp}

            _append_to_csv(row, results_file)
            all_results.append(row)

        print(f"\nDone. Results saved to {results_file}")
        return all_results

    @staticmethod
    def load_results(results_file='results/grid_results.csv'):
        """Load results CSV as a pandas DataFrame.

        Usage::

            df = ExperimentGrid.load_results('results/my_grid.csv')
            print(df[['dataset', 'model', 'lr', 'test_mae', 'test_rmse']].to_string())
        """
        import pandas as pd
        df = pd.read_csv(results_file)
        # Put identifier columns first for readability
        id_cols = ['dataset', 'model', 'status', 'timestamp']
        metric_cols = [c for c in df.columns if c.startswith(('val_', 'test_'))]
        other_cols = [c for c in df.columns
                      if c not in id_cols and c not in metric_cols]
        return df[id_cols + metric_cols + other_cols]
