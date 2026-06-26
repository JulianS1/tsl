import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from tsl import logger
from tsl.data import SpatioTemporalDataModule, SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.engines import Predictor
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.utils.casting import torch_to_numpy

from AIRU_Experiments.setup import Setup

DATASETS = ['la', 'bay', 'pems3', 'pems4', 'pems7', 'pems8', 'gpvar']
MODELS = ['gwnet', 'dcrnn', 'fcrnn', 'tcn','gru_diffconv']

CONNECTIVITY = {
    'la':    {'method': 'distance', 'threshold': 0.1, 'include_self': False},
    'bay':   {'method': 'distance', 'threshold': 0.1, 'include_self': False},
    'pems3': {'method': 'binary', 'include_self': False},
    'pems4': {'method': 'binary', 'include_self': False},
    'pems7': {'method': 'binary', 'include_self': False},
    'pems8': {'method': 'binary', 'include_self': False},
    'gpvar': {},  # uses the graph built into the dataset; no extra kwargs needed
}

SPLITS = {
    'la':    {'val_len': 0.1, 'test_len': 0.2},
    'bay':   {'val_len': 0.1, 'test_len': 0.2},
    'pems3': {'val_len': 0.2, 'test_len': 0.2},
    'pems4': {'val_len': 0.2, 'test_len': 0.2},
    'pems7': {'val_len': 0.2, 'test_len': 0.2},
    'pems8': {'val_len': 0.2, 'test_len': 0.2},
    'gpvar': {'val_len': 0.1, 'test_len': 0.2},
}


def resolve_accelerator(requested: str) -> str:
    if requested == 'cpu':
        return 'cpu'
    if not torch.cuda.is_available():
        if requested == 'gpu':
            logger.warning("GPU requested but CUDA is not available; using CPU.")
        return 'cpu'
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    sm_tag = f"sm_{major}{minor}"
    if sm_tag not in torch.cuda.get_arch_list():
        logger.warning(
            f"GPU '{torch.cuda.get_device_name()}' ({sm_tag}) is not supported "
            f"by this PyTorch build. Falling back to CPU."
        )
        return 'cpu'
    return 'gpu'


def parse_args():
    p = argparse.ArgumentParser(
        description='Train a TSL model on a traffic dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--dataset', default='la', choices=DATASETS)
    p.add_argument('--model', default='gwnet', choices=MODELS)
    p.add_argument('--connectivity-layout', default='edge_index',
                   choices=['edge_index', 'csr', 'dense'])

    p.add_argument('--window', type=int, default=12)
    p.add_argument('--horizon', type=int, default=12)
    p.add_argument('--stride', type=int, default=1)

    # Training params
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--lr-decay', type=float, default=0.97,
                   help='ExponentialLR gamma per epoch (paper=0.97). '
                        'Set 1.0 to disable.')
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--workers', type=int, default=119)
    p.add_argument('--log-dir', default='logs/experiment')
    p.add_argument('--logger-backend', default=None,
                   choices=[None, 'tensorboard', 'wandb'])
    p.add_argument('--accelerator', default='auto',
                   choices=['auto', 'gpu', 'cpu'])
    p.add_argument('--results-file', default=None,
                   help='CSV file to append results to (optional).')

    # Model hyperparameters — unused ones are stripped by filter_model_args_
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

    return p.parse_args()


def run_experiment(args):
    """Run a single experiment. Returns a dict of val/test metrics."""

    setup = Setup(args)

    # Dataset & adjacency matrix
    dataset = setup.get_dataset(args.dataset)
    logger.info(f"Loaded '{args.dataset}': {dataset.n_nodes} nodes, "
                f"{len(dataset.index)} timesteps")

    connectivity_kwargs = dict(CONNECTIVITY[args.dataset],
                               layout=args.connectivity_layout)
    adj = dataset.get_connectivity(**connectivity_kwargs)

    # Covariates (time-of-day sine/cosine encoding)
    covariates = {}
    try:
        covariates['u'] = dataset.datetime_encoded('day').values
    except Exception:
        pass  # dataset has no datetime index

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
    splitter = dataset.get_splitter(**SPLITS[args.dataset])

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=splitter,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    dm.setup()

    # Model & predictor
    model_cls = setup.get_model(args.model)

    exog_size = (torch_dataset.input_map.u.shape[-1]
                 if 'u' in torch_dataset.input_map else 0)

    # Start with data-derived sizes, then add all model hparams.
    # filter_model_args_ removes any that the chosen model doesn't accept.
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
    )
    model_cls.filter_model_args_(model_kwargs)

    loss_fn = torch_metrics.MaskedMAE()
    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mape': torch_metrics.MaskedMAPE(),
        'mae_at_15': torch_metrics.MaskedMAE(at=2),
        'mae_at_30': torch_metrics.MaskedMAE(at=5),
        'mae_at_60': torch_metrics.MaskedMAE(at=11),
        'mse_at_15': torch_metrics.MaskedMSE(at=2),
        'mse_at_30': torch_metrics.MaskedMSE(at=5),
        'mse_at_60': torch_metrics.MaskedMSE(at=11),
        'mape_at_15': torch_metrics.MaskedMAPE(at=2),
        'mape_at_30': torch_metrics.MaskedMAPE(at=5),
        'mape_at_60': torch_metrics.MaskedMAPE(at=11),
    }

    use_scheduler = hasattr(args, 'lr_decay') and args.lr_decay < 1.0
    predictor = Predictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr, 'weight_decay': args.weight_decay},
        loss_fn=loss_fn,
        metrics=log_metrics,
        scale_target=False,
        scheduler_class=torch.optim.lr_scheduler.ExponentialLR if use_scheduler else None,
        scheduler_kwargs={'gamma': args.lr_decay} if use_scheduler else None,
    )

    #Train
    callbacks = [
        EarlyStopping(monitor='val_mae', patience=args.patience, mode='min'),
        ModelCheckpoint(dirpath=args.log_dir, save_top_k=1,
                        monitor='val_mae', mode='min'),
    ]

    accelerator = resolve_accelerator(args.accelerator)
    if accelerator == 'gpu':
        torch.set_float32_matmul_precision('high')

    exp_logger = setup.get_logger(args.logger_backend)

    trainer = Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.log_dir,
        logger=exp_logger,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
    )

    trainer.fit(predictor, datamodule=dm)

    #Test on best model checkpoint
    predictor.load_model(callbacks[1].best_model_path)
    predictor.freeze()
    trainer.test(predictor, datamodule=dm)

    def evaluate(dataloader, split):
        out = trainer.predict(predictor, dataloaders=dataloader)
        out = predictor.collate_prediction_outputs(out)
        out = torch_to_numpy(out)
        y_hat, y_true, mask = out['y_hat'], out['y'], out.get('mask')
        return {
            f'{split}_mae':  numpy_metrics.mae(y_hat, y_true, mask),
            f'{split}_rmse': numpy_metrics.rmse(y_hat, y_true, mask),
            f'{split}_mape': numpy_metrics.mape(y_hat, y_true, mask),
        }

    res = {}
    res.update(evaluate(dm.val_dataloader(), 'val'))
    res.update(evaluate(dm.test_dataloader(), 'test'))

    logger.info("=" * 50)
    for k, v in res.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 50)

    if args.results_file:
        from AIRU_Experiments.grid import _append_to_csv
        row = {**vars(args), **res, 'status': 'ok'}
        _append_to_csv(row, args.results_file)

    return res


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
