"""Train Graph WaveNet on any TSL dataset.

Usage (from repo root or any directory):
    conda run -n tsl python examples/forecasting/run_gwnet.py --dataset la
    conda run -n tsl python examples/forecasting/run_gwnet.py --dataset pems4 --epochs 100
    conda run -n tsl python examples/forecasting/run_gwnet.py --dataset air_quality

Available datasets:
    Traffic:      la, bay, pems3, pems4, pems7, pems8
    Benchmarks:   electricity, exchange, solar, traffic_bench
    Other:        air_quality, elergone, engrad, largeST, pvus, gpvar, gpvar_az
"""

import argparse
import sys
from pathlib import Path

# Make 'tsl' importable when this script is run directly (not as part of an
# installed package). Adds the repo root (two levels up) to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from tsl import logger
from tsl.data import SpatioTemporalDataModule, SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.engines import Predictor
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.nn.models import GraphWaveNetModel
from tsl.utils.casting import torch_to_numpy

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_INFO = {
    # name: (class_path, constructor_kwargs, connectivity_kwargs,
    #         splitter_kwargs, has_datetime_covariates)
    'la': {
        'class': 'MetrLA',
        'ctor': {'impute_zeros': True},
        'connectivity': {'method': 'distance', 'threshold': 0.1,
                         'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'bay': {
        'class': 'PemsBay',
        'ctor': {},
        'connectivity': {'method': 'distance', 'threshold': 0.1,
                         'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'pems3': {
        'class': 'PeMS03',
        'ctor': {},
        'connectivity': {'method': 'binary', 'include_self': False},
        'splitter': {'val_len': 0.2, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'pems4': {
        'class': 'PeMS04',
        'ctor': {},
        'connectivity': {'method': 'binary', 'include_self': False},
        'splitter': {'val_len': 0.2, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'pems7': {
        'class': 'PeMS07',
        'ctor': {},
        'connectivity': {'method': 'binary', 'include_self': False},
        'splitter': {'val_len': 0.2, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'pems8': {
        'class': 'PeMS08',
        'ctor': {},
        'connectivity': {'method': 'binary', 'include_self': False},
        'splitter': {'val_len': 0.2, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'air_quality': {
        'class': 'AirQuality',
        'ctor': {'impute_nans': True},
        'connectivity': {'method': 'distance', 'threshold': 0.1,
                         'include_self': False},
        'splitter': {},  # uses AirQuality's own splitter (month-based)
        'datetime_cov': 'day',
    },
    'elergone': {
        'class': 'Elergone',
        'ctor': {},
        'connectivity': {'method': 'full', 'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'engrad': {
        'class': 'EngRad',
        'ctor': {},
        'connectivity': {'method': 'full', 'include_self': False},
        'splitter': {},
        'datetime_cov': 'day',
    },
    'electricity': {
        'class': 'ElectricityBenchmark',
        'ctor': {},
        'connectivity': {'method': 'full', 'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'exchange': {
        'class': 'ExchangeBenchmark',
        'ctor': {},
        'connectivity': {'method': 'full', 'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'solar': {
        'class': 'SolarBenchmark',
        'ctor': {},
        'connectivity': {'method': 'full', 'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'traffic_bench': {
        'class': 'TrafficBenchmark',
        'ctor': {},
        'connectivity': {'method': 'full', 'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'largeST': {
        'class': 'LargeST',
        'ctor': {},
        'connectivity': {'method': 'distance', 'threshold': 0.1,
                         'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'pvus': {
        'class': 'PvUS',
        'ctor': {},
        'connectivity': {'method': 'distance', 'threshold': 0.1,
                         'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': 'day',
    },
    'gpvar': {
        'class': 'GPVARDataset',
        'ctor': {},
        'connectivity': {'method': 'full', 'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': None,
    },
    'gpvar_az': {
        'class': 'GPVARDatasetAZ',
        'ctor': {},
        'connectivity': {'method': 'full', 'include_self': False},
        'splitter': {'val_len': 0.1, 'test_len': 0.2},
        'datetime_cov': None,
    },
}


def import_dataset_class(class_name: str):
    import tsl.datasets as ds_module
    try:
        return getattr(ds_module, class_name)
    except AttributeError:
        raise ValueError(f"Dataset class '{class_name}' not found in tsl.datasets.")


def build_dataset(info: dict, connectivity_layout: str):
    cls = import_dataset_class(info['class'])
    dataset = cls(**info['ctor'])
    connectivity_kwargs = dict(info['connectivity'], layout=connectivity_layout)
    adj = dataset.get_connectivity(**connectivity_kwargs)
    return dataset, adj


def build_splitter(dataset, info: dict):
    splitter_kwargs = info.get('splitter', {})
    return dataset.get_splitter(**splitter_kwargs)


def resolve_accelerator(requested: str) -> str:
    """Return 'gpu' or 'cpu', falling back to CPU when the GPU's CUDA
    capability is not supported by the installed PyTorch build."""
    if requested == 'cpu':
        return 'cpu'
    if not torch.cuda.is_available():
        if requested == 'gpu':
            logger.warning("GPU requested but CUDA is not available; using CPU.")
        return 'cpu'
    # Check that the device's SM version is among those compiled into this build.
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    supported = torch.cuda.get_arch_list()  # e.g. ['sm_50', 'sm_70', ...]
    sm_tag = f"sm_{major}{minor}"
    if sm_tag not in supported:
        logger.warning(
            f"GPU '{torch.cuda.get_device_name(device)}' has CUDA capability "
            f"{sm_tag}, which is not in the current PyTorch build's arch list "
            f"({supported}). Falling back to CPU.\n"
            "To use this GPU, install a PyTorch build that supports it:\n"
            "  https://pytorch.org/get-started/locally/"
        )
        return 'cpu'
    return 'gpu'


def parse_args():
    p = argparse.ArgumentParser(
        description='Train Graph WaveNet on a TSL dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    p.add_argument('--dataset', default='la',
                   choices=sorted(DATASET_INFO.keys()),
                   help='Dataset to use.')
    p.add_argument('--connectivity-layout', default='edge_index',
                   choices=['edge_index', 'csr', 'dense'],
                   help='Sparse format for the adjacency matrix.')

    # Windowing
    p.add_argument('--window', type=int, default=12,
                   help='Input sequence length (number of past steps).')
    p.add_argument('--horizon', type=int, default=12,
                   help='Forecast horizon (number of future steps).')
    p.add_argument('--stride', type=int, default=1,
                   help='Stride between successive windows.')

    # Training
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--patience', type=int, default=50,
                   help='Early-stopping patience (val_mae).')
    p.add_argument('--workers', type=int, default=0,
                   help='DataLoader workers.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-dir', default='logs/gwnet',
                   help='Directory for checkpoints and TensorBoard logs.')

    # GWN hyperparameters
    p.add_argument('--hidden-size', type=int, default=32)
    p.add_argument('--ff-size', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=8)
    p.add_argument('--emb-size', type=int, default=10)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--temporal-kernel-size', type=int, default=2)
    p.add_argument('--spatial-kernel-size', type=int, default=2)
    p.add_argument('--dilation', type=int, default=2)
    p.add_argument('--dilation-mod', type=int, default=2)
    p.add_argument('--norm', default='batch', choices=['batch', 'layer', 'none'])
    p.add_argument('--no-learned-adjacency', action='store_true',
                   help='Disable the adaptive adjacency matrix.')
    p.add_argument('--accelerator', default='auto',
                   choices=['auto', 'gpu', 'cpu'],
                   help='Hardware accelerator. "auto" uses GPU if available '
                        'and falls back to CPU when the GPU architecture is '
                        'unsupported by the current PyTorch build.')

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    info = DATASET_INFO[args.dataset]

    # -------------------------------------------------------------------------
    # Dataset & adjacency
    # -------------------------------------------------------------------------
    logger.info(f"Loading dataset '{args.dataset}' ({info['class']})...")
    dataset, adj = build_dataset(info, args.connectivity_layout)
    logger.info(f"  Nodes: {dataset.n_nodes} | Steps: {len(dataset.index)}")

    # -------------------------------------------------------------------------
    # Covariates
    # -------------------------------------------------------------------------
    covariates = {}
    if info['datetime_cov'] is not None:
        try:
            covariates['u'] = dataset.datetime_encoded(info['datetime_cov']).values
        except Exception:
            pass  # dataset has no datetime index — skip covariate

    # -------------------------------------------------------------------------
    # SpatioTemporalDataset
    # -------------------------------------------------------------------------
    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        mask=dataset.mask if dataset.has_mask else None,
        connectivity=adj,
        covariates=covariates if covariates else None,
        horizon=args.horizon,
        window=args.window,
        stride=args.stride,
    )

    transform = {'target': StandardScaler(axis=(0, 1))}

    splitter = build_splitter(dataset, info)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=splitter,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    dm.setup()

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    exog_size = (torch_dataset.input_map.u.shape[-1]
                 if 'u' in torch_dataset.input_map else 0)

    model_kwargs = dict(
        n_nodes=torch_dataset.n_nodes,
        input_size=torch_dataset.n_channels,
        output_size=torch_dataset.n_channels,
        horizon=torch_dataset.horizon,
        exog_size=exog_size,
        hidden_size=args.hidden_size,
        ff_size=args.ff_size,
        n_layers=args.n_layers,
        emb_size=args.emb_size,
        dropout=args.dropout,
        temporal_kernel_size=args.temporal_kernel_size,
        spatial_kernel_size=args.spatial_kernel_size,
        dilation=args.dilation,
        dilation_mod=args.dilation_mod,
        norm=args.norm,
        learned_adjacency=not args.no_learned_adjacency,
    )
    GraphWaveNetModel.filter_model_args_(model_kwargs)

    loss_fn = torch_metrics.MaskedMAE()
    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mape': torch_metrics.MaskedMAPE(),
    }

    predictor = Predictor(
        model_class=GraphWaveNetModel,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr, 'weight_decay': args.weight_decay},
        loss_fn=loss_fn,
        metrics=log_metrics,
        scale_target=False,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    callbacks = [
        EarlyStopping(monitor='val_mae', patience=args.patience, mode='min'),
        ModelCheckpoint(dirpath=args.log_dir, save_top_k=1,
                        monitor='val_mae', mode='min'),
    ]

    accelerator = resolve_accelerator(args.accelerator)
    if accelerator == 'gpu':
        torch.set_float32_matmul_precision('high')

    trainer = Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.log_dir,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
    )

    logger.info("Training...")
    trainer.fit(predictor, datamodule=dm)

    # -------------------------------------------------------------------------
    # Testing
    # -------------------------------------------------------------------------
    ckpt_path = callbacks[1].best_model_path
    predictor.load_model(ckpt_path)
    predictor.freeze()

    trainer.test(predictor, datamodule=dm)

    def evaluate(dataloader, split):
        out = trainer.predict(predictor, dataloaders=dataloader)
        out = predictor.collate_prediction_outputs(out)
        out = torch_to_numpy(out)
        y_hat, y_true, mask = out['y_hat'], out['y'], out.get('mask')
        return {
            f'{split}_mae': numpy_metrics.mae(y_hat, y_true, mask),
            f'{split}_rmse': numpy_metrics.rmse(y_hat, y_true, mask),
            f'{split}_mape': numpy_metrics.mape(y_hat, y_true, mask),
        }

    results = {}
    results.update(evaluate(dm.val_dataloader(), 'val'))
    results.update(evaluate(dm.test_dataloader(), 'test'))

    logger.info("=" * 50)
    logger.info("Results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 50)

    return results


if __name__ == '__main__':
    main()
