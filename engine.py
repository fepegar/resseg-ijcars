from pathlib import Path
from typing import Optional

import torch
import numpy as np
import pandas as pd
import torchio as tio
from tqdm import tqdm

import loss
import utils
import models
import visualization


ORANGE = '#cf8415'
GREEN = '#68ba16'
BLUE = '#167eba'
PURPLE = '#a869c2'


def run_epoch(
        loader,
        model,
        *,
        train,
        criterion,
        scaler,
        num_batches,
        num_instances,
        num_layers_finetune=None,
        pseudo_loader=None,
        tb_log=None,
        num_iterations: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        plot_every: int = 50,
        **kwargs,
        ):
    model.train(train)
    models.freeze_except(model, num_layers_finetune)  # noop if num_layers_finetune is None
    epoch_losses = []
    progress = tqdm(loader, unit='batch', **kwargs)
    pseudo_iterator = None if pseudo_loader is None else iter(pseudo_loader)
    for i, batch in enumerate(progress):
        batch['name'] = batch['image']['stem']  # for TensorBoard?
        batch_pseudo = get_batch_pseudo(pseudo_loader, pseudo_iterator)
        batch = add_pseudolabeled(batch, batch_pseudo)
        batch_losses_array, outputs = get_losses(
            batch, train, model, criterion, scaler, optimizer, num_batches, num_instances, tb_log)
        batch_loss_mean = batch_losses_array.mean()
        if np.isnan(batch_loss_mean):
            import warnings
            path = '/tmp/batch_nan.pth'
            if hasattr(model, 'module'):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            d = dict(
                batch=batch,
                train=train,
                model=model_state,
                criterion=criterion,
                scaler=scaler.state_dict(),
                optimizer=optimizer.state_dict(),
                num_batches=num_batches,
                num_instances=num_instances,
            )
            warnings.warn(f'NaN loss detected. Saving to {path}')
            torch.save(d, path)
        progress.set_postfix(batch_loss=batch_loss_mean)
        batch_losses_flat = batch_losses_array.mean(axis=1)  # mean of bg and fg
        epoch_losses.extend(batch_losses_flat.tolist())
        last_full_batch = i == len(loader) - 2  # in case last batch is incomplete
        single_batch = len(loader) == 1
        if last_full_batch or single_batch:
            image = visualization.get_batch_grid(batch, outputs)
            tb_log.add_image(f'epoch/{kwargs["desc"]}', image, num_instances)
    epoch_losses_array = np.array(epoch_losses)
    tb_log.add_scalars('loss', {f'epoch/{kwargs["desc"]}': epoch_losses_array.mean()}, num_instances)
    return epoch_losses_array


def get_batch_pseudo(pseudo_loader, pseudo_iterator):
    if pseudo_loader is None:
        return None
    try:
        batch = next(pseudo_iterator)
    except StopIteration:
        pseudo_iterator = iter(pseudo_loader)
        batch = next(pseudo_iterator)
    return batch


def add_pseudolabeled(batch, batch_pseudo):
    if batch_pseudo is None:
        result = batch
    else:
        del batch['random_resection']  # lazy but effective
        batch_pseudo['name'] = batch_pseudo['image']['stem']  # for TensorBoard?
        result = collate_batches([batch, batch_pseudo])
    return result


def get_losses(batch, train, model, criterion, scaler, optimizer, num_batches, num_instances, tb_log):
    inputs, targets = prepare_batch(batch, one_hot=True)
    with torch.set_grad_enabled(train):
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(inputs).softmax(dim=1)
            batch_losses = criterion(outputs, targets)
            batch_loss = batch_losses.mean()
        batch_losses_array = batch_losses.cpu().detach().numpy()
        if not train:
            return batch_losses_array, outputs
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        iter_plot = num_batches == 0 or num_batches % 10 == 1
        if iter_plot:
            image = visualization.get_batch_grid(batch, outputs)
            tb_log.add_image('batch/Training', image, num_instances)
        tb_log.add_scalars('loss', {'batch/training': batch_losses_array.mean()}, num_instances)
        num_batches += 1
        num_instances += len(batch)
    return batch_losses_array, outputs


def prepare_batch(batch, *, one_hot):
    x = batch['image']['data'].float()
    y = batch['label']['data'].float()
    if one_hot:
        fg = y
        bg = 1 - fg
        y = torch.cat((bg, fg), dim=1)
    device = utils.get_device()
    x = x.to(device)
    y = y.to(device)
    return x, y


class Evaluator:
    def __init__(self):
        pass

    def infer(self, model, loader, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        records = []
        model.eval()
        device = utils.get_device()
        with torch.no_grad():
            for batch in tqdm(loader):
                inputs = batch['image'][tio.DATA].float().to(device)
                segs = model(inputs).softmax(dim=1)[:, 1:].cpu() > 0.5
                affines = batch['image'][tio.AFFINE]
                paths = batch['image'][tio.PATH]
                targets = batch['label'][tio.DATA]
                for seg, target, affine, path in zip(segs, targets, affines, paths):
                    image = tio.LabelMap(tensor=seg, affine=affine.numpy())
                    path = Path(path)
                    out_path = out_dir / path.name.replace('.nii', '_seg_cnn.nii')
                    image.save(out_path)
                    confusion = loss.get_confusion(seg[0].float(), target[0].float())
                    precision = loss.get_precision(confusion)
                    recall = loss.get_recall(confusion)
                    dice = loss.get_dice_from_precision_and_recall(precision, recall)
                    sid = path.name.split('_t1_post')[0] if '_t1_post' in path.name else path.name.split('.')[0]
                    record = dict(
                        Subject=sid,
                        Precision=precision.item(),
                        Recall=recall.item(),
                        Dice=dice.item(),
                    )
                    records.append(record)
        df = pd.DataFrame.from_records(records)
        return df


# From PyTorch
# This version uses cat instead of stack
def collate_batches(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    from torch._six import container_abcs, string_classes, int_classes

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return collate_batches([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_batches([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_batches(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # We shouldn't need this here because we replaced stack with cat
        # # check to make sure that the elements in batch have consistent size
        # it = iter(batch)
        # elem_size = len(next(it))
        # if not all(len(elem) == elem_size for elem in it):
        #     raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_batches(samples) for samples in transposed]

    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}"
    )

    raise TypeError(default_collate_err_msg_format.format(elem_type))
