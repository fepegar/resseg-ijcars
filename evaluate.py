import sys
import click
from pathlib import Path


@click.command()
@click.argument('image-dir', type=click.Path(exists=True))
@click.argument('label-dir', type=click.Path(exists=True))
@click.argument('checkpoint-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output-dir', type=click.Path())
@click.argument('landmarks-path', type=click.Path(dir_okay=False))
@click.argument('df-path', type=click.Path(dir_okay=False))
@click.option('--batch-size', '-b', type=int, default=6, show_default=True)
@click.option('--num-workers', '-j', type=int, default=12, show_default=True)
@click.option('--multi-gpu/--single-gpu', '-m', default=True)
def main(image_dir, label_dir, checkpoint_path, output_dir, landmarks_path, df_path, batch_size, num_workers, multi_gpu):
    import torch
    import torchio as tio
    import models
    import datasets
    import engine
    import utils

    fps = get_paths(image_dir)
    lfps = get_paths(label_dir)
    assert len(fps) == len(lfps)
    # key must be 'image' as in get_test_transform
    subjects = [tio.Subject(image=tio.ScalarImage(fp), label=tio.LabelMap(lfp)) for (fp, lfp) in zip(fps, lfps)]
    transform = datasets.get_test_transform(landmarks_path)
    dataset = tio.SubjectsDataset(subjects, transform)
    checkpoint = torch.load(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.get_unet().to(device)
    if multi_gpu:
        model = torch.nn.DataParallel(model)
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    output_dir = Path(output_dir)
    model.eval()
    torch.set_grad_enabled(False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    output_dir.mkdir(parents=True)
    evaluator = engine.Evaluator()
    df = evaluator.infer(model, loader, output_dir)
    df.to_csv(df_path)
    med, iqr = 100 * utils.get_median_iqr(df.Dice)
    print(f'{med:.1f} ({iqr:.1f})')
    return 0


def get_paths(folder):
    import utils
    folder = Path(folder)
    if folder.is_file():
        fps = [folder]
    elif folder.is_dir():
        fps = utils.sglob(folder)
    return fps


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
