"""
From experience:
- If hangs using DataParallel, kill process (kill -p [pid]) and use smaller batch size
- If "RuntimeError: input tensor must fit into 32-bit index math", use smaller batch size
"""

from pathlib import Path

import torch
import numpy as np
from tqdm import trange
from sacred import Experiment

import loss
import utils
import models
import engine
import datasets
import observers

ex = Experiment()
file_observer = observers.add_file_storage_observer(ex)
slack_observer = observers.add_slack_observer(ex)
runs_dir = Path(file_observer.basedir)


# flake8: noqa: F841
@ex.config
def config():
    seed = 42
    datasets_dir = str(Path('~/datasets').expanduser())
    real_dataset_dir = str(Path('~/datasets/real/london').expanduser())
    debug = False


@ex.config
def config_resources():
    train_batch_size_per_gpu = 5  # 5 with DataParallel take up to 32131 and 29217 MB
    num_workers = 12  # seems to be the fastest number
    multi_gpu = True
    cpu = False
    cache_val_set = True
    use_amp = True
    debug_ratio = 0.02
    validate = True


@ex.config
def config_training():
    num_epochs = 60
    learning_rate = 1e-3
    scheduler_step_size = 20  # epochs
    pre_trained_checkpoint_path = None
    num_layers_finetune = None  # None means all
    load_optimizer = True
    load_scheduler = True
    continue_iterations = True
    pseudo_dirname = None
    augment = True
    histogram_standardization = True


@ex.config
def config_semisupervised():
    threshold_pseudo = 0.2
    percentile_pseudo = None  # 50
    metric = 'QCD'
    summary_path = None
    dataset_name = 'pseudo'
    use_pseudolabeled = False
    pseudolabels_batch_size_per_gpu = 1


@ex.config
def config_cross_validation():
    num_folds = 0


@ex.capture
def get_resection_params(white_matter_p, blood_clot_p, shape, texture):
    return {
        'wm_lesion_p': white_matter_p,
        'clot_p': blood_clot_p,
        'shape': shape,
        'texture': texture,
    }


@ex.capture
def get_data_module_public(
        datasets_dir,
        real_dataset_dir,
        train_batch_size_per_gpu,
        num_workers,
        debug,
        debug_ratio,
        augment,
        multi_gpu,
        use_pseudolabeled,
        pseudolabels_batch_size_per_gpu,
        histogram_standardization,
        _log,
        ):
    num_devices = torch.cuda.device_count() if multi_gpu else 1
    train_batch_size = num_devices * train_batch_size_per_gpu
    if use_pseudolabeled:
        pseudolabels_batch_size = num_devices * pseudolabels_batch_size_per_gpu
        train_batch_size -= pseudolabels_batch_size  # leave room in batch for pseudolabeled images
    return datasets.DataModulePublic(
        datasets_dir,
        real_dataset_dir,
        get_resection_params(),  # pylint: disable=E1120
        train_batch_size,
        num_workers,
        debug=debug,
        debug_ratio=debug_ratio,
        log=_log,
        verbose=True,
        augment=augment,
        histogram_standardization=histogram_standardization,
    )


@ex.capture
def get_data_module_real(
        fold,
        num_folds,
        datasets_dir,
        dataset_name,
        train_batch_size_per_gpu,
        num_workers,
        multi_gpu,
        pseudo_dirname,
        _log,
        use_public_landmarks,
        ):
    assert dataset_name is not None
    num_devices = torch.cuda.device_count() if multi_gpu else 1
    train_batch_size = num_devices * train_batch_size_per_gpu
    return datasets.DataModuleCV(
        fold,
        num_folds,
        datasets_dir,
        dataset_name,
        train_batch_size,
        num_workers,
        log=_log,
        verbose=True,
        pseudo_dirname=pseudo_dirname,
        use_public_landmarks=use_public_landmarks,
    )


@ex.capture
def setup_model(model, multi_gpu, cpu):  # gpus ignored for now
    if cpu:
        utils.get_device = lambda: 'cpu'
    else:
        model.to(utils.get_device())
        if multi_gpu:
            # model = DDP(model)
            model = torch.nn.DataParallel(model)
    return model


@ex.capture
def get_experiment_id(_run):
    # pylint: disable=protected-access
    experiment_id = _run._id
    if experiment_id is None:
        experiment_id = 'unobserved'
    return experiment_id


def get_experiment_dir():
    experiment_id = get_experiment_id()  # pylint: disable=E1120
    experiment_dir = runs_dir / experiment_id
    return experiment_dir


@ex.capture
def setup_benchmark(debug):
    torch.backends.cudnn.benchmark = not debug


class Trainer:
    def __init__(self, fold=None):
        self.num_epochs = None
        self.num_processed_instances = np.array(0)  # array so it can be "passed by reference"
        self.num_processed_batches = np.array(0)  # array so it can be "passed by reference"
        self.fold = fold
        if fold is None:
            fold_string = ''
            self.data = get_data_module_public()  # pylint: disable=E1120
        else:
            fold_string = f'_fold_{fold}'
            self.data = get_data_module_real(fold)  # pylint: disable=E1120
        self.pseudo_loader = self.get_pseudo_loader()
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()  # pylint: disable=E1120
        self.scheduler = self.get_scheduler()  # pylint: disable=E1120
        self.load_pretrained()
        self.scaler = self.get_scaler()  # pylint: disable=E1120
        self.criterion = loss.DiceLoss()
        self.experiment_dir = get_experiment_dir()  # pylint: disable=E1120
        self.best_checkpoint_path = self.experiment_dir / f'checkpoint{fold_string}.pth'
        self.last_checkpoint_path = self.experiment_dir / f'checkpoint_last{fold_string}.pth'
        self.evaluation_path = self.experiment_dir / f'evaluation{fold_string}.csv'
        self.best_val_loss = 1
        self.inference_dir = self.experiment_dir / 'inference'
        self.tensorboard_dir = self.experiment_dir / f'tensorboard{fold_string}'
        self.tb_log = self.get_tb_writer()
        setup_benchmark()

    @ex.capture
    def load_pretrained(
            self,
            pre_trained_checkpoint_path,
            load_optimizer,
            load_scheduler,
            continue_iterations,
            num_layers_finetune,
            _log,
            ):
        if pre_trained_checkpoint_path is None:
            return
        checkpoint = torch.load(pre_trained_checkpoint_path)
        if hasattr(self.model, 'module'):  # DataParallel
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if load_scheduler:
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                _log.warning('Scheduler not found in checkpoint')
        if continue_iterations and 'num_processed_batches' in checkpoint:
            self.num_processed_batches = checkpoint['num_processed_batches']
            self.num_processed_instances = checkpoint['num_processed_instances']
            self.num_epochs = checkpoint['epoch']
        _log.info(f'Number of processed batches: {self.num_processed_batches}')
        _log.info(f'Number of processed instances: {self.num_processed_instances}')

    def get_tb_writer(self):
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=self.tensorboard_dir)

    def get_model(self):
        return setup_model(models.get_unet())  # pylint: disable=E1120

    @ex.capture
    def get_optimizer(self, learning_rate):
        optimizer = torch.optim.AdamW(self.model.parameters(), learning_rate)
        return optimizer

    @ex.capture
    def get_scheduler(self, scheduler_step_size):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size)
        return scheduler

    @ex.capture
    def get_scaler(self, use_amp):
        if not torch.cuda.is_available() or not next(self.model.parameters()).is_cuda:
            use_amp = False
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        return scaler

    @ex.capture
    def get_pseudo_loader(
            self,
            threshold_pseudo,
            percentile_pseudo,
            metric,
            summary_path,
            dataset_name,
            use_pseudolabeled,
            num_workers,
            _log,
            ):
        if not use_pseudolabeled:
            loader = None
        else:
            if threshold_pseudo is None and percentile_pseudo is None:
                _log.warning('No threshold or percentile for pseudo. Using percentile 100')
                percentile_pseudo = 100
            if threshold_pseudo is not None and percentile_pseudo is not None:
                raise RuntimeError('Only threshold or percentile for pseudo should be passed')
            loader = datasets.get_pseudo_loader(
                threshold_pseudo,
                percentile_pseudo,
                metric,
                summary_path,
                dataset_name,
                num_workers,
            )
            _log.info(f'{len(loader.dataset):4} pseudolabeled instances')
        return loader

    def save_state(self, loss, epoch, path):
        if hasattr(self.model, 'module'):  # DataParallel
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'num_processed_batches': self.num_processed_batches,
            'num_processed_instances': self.num_processed_instances,
        }
        torch.save(checkpoint, path)

    def check_best_loss(self, loss, epoch):
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            self.save_state(loss, epoch, self.best_checkpoint_path)

    @ex.capture
    def run_validation(self, epoch, seed, cache_val_set, _log):
        # We always want validation to be the same
        with torch.random.fork_rng([]):
            # Using a cached validation set will make things much faster, hopefully
            if not cache_val_set:
                torch.manual_seed(seed)
            losses = engine.run_epoch(
                self.data.val_loader,
                self.model,
                train=False,
                criterion=self.criterion,
                scaler=self.scaler,
                num_batches=self.num_processed_batches,
                num_instances=self.num_processed_instances,
                tb_log=self.tb_log,
                colour=engine.GREEN,
                desc='Validation',
            )
        # _log.info(f'{"Mean val loss:":20}{losses.mean():.3f}')
        if epoch > 0:
            self.check_best_loss(losses.mean(), epoch)
        self.last_checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
        self.save_state(losses.mean(), epoch, self.last_checkpoint_path)
        return losses

    @ex.capture
    def run_train(self, _log, num_layers_finetune):
        losses = engine.run_epoch(
            self.data.train_loader,
            self.model,
            train=True,
            criterion=self.criterion,
            scaler=self.scaler,
            num_batches=self.num_processed_batches,
            num_instances=self.num_processed_instances,
            optimizer=self.optimizer,
            tb_log=self.tb_log,
            num_layers_finetune=num_layers_finetune,
            pseudo_loader=self.pseudo_loader,
            colour=engine.BLUE,
            desc='Training  ',
        )
        self.scheduler.step()
        return losses

    @ex.capture
    def train(self, num_epochs, validate):
        progress = trange(num_epochs, desc='Fit       ', unit='epoch')
        if validate:
            val_losses = self.run_validation(epoch=-1)  # pylint: disable=E1120
            progress.set_postfix(
                val=val_losses.mean(),
            )
        for epoch in progress:
            train_losses = self.run_train()
            postfix = {'train': train_losses.mean()}
            if validate:
                val_losses = self.run_validation(epoch=epoch)
                postfix['val'] = val_losses.mean()
            progress.set_postfix(**postfix)

    @ex.capture
    def evaluate(self, _log, load_best=True):
        evaluator = engine.Evaluator()
        if load_best:
            if self.best_checkpoint_path.is_file():
                checkpoint = torch.load(self.best_checkpoint_path)
                if hasattr(self.model, 'module'):  # DataParallel
                    self.model.module.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint['model'])
            else:
                _log.warning(f'Checkpoint not found: {self.best_checkpoint_path}')
        df = evaluator.infer(
            self.model,
            self.data.test_loader,
            self.inference_dir,
        )
        df.to_csv(self.evaluation_path)
        dices = df.Dice.values.tolist()
        return dices

    @ex.capture
    def run(self, num_epochs, _log):
        if self.num_epochs is not None:
            num_epochs = num_epochs - self.num_epochs
            _log.info(f'Continuing for {num_epochs} more epochs')
        self.train(num_epochs)
        return self.evaluate()


@ex.automain
def run(num_epochs, debug, num_folds, seed):
    torch.manual_seed(seed)
    if num_folds == 0:
        dices = Trainer().run(num_epochs)  # pylint: disable=E1120
    else:
        progress = trange(
            1 if debug else num_folds,
            unit='fold',
            desc='Cross-val.',
            colour=engine.PURPLE,
        )
        dices = []
        for fold in progress:
            results = Trainer(fold=fold).run(num_epochs)  # pylint: disable=E1120
            dices.extend(results)
    median, iqr = 100 * utils.get_median_iqr(dices)
    results_summary_path = runs_dir / 'results.txt'
    with open(results_summary_path, 'a') as f:
        line = f'{get_experiment_id():3}: {median:.3f} ({iqr:.3f})\n'
        f.write(line)
        print(results_summary_path, 'updated:', line)
    return median, iqr
