import sys
import click
from pathlib import Path


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
@click.argument('checkpoint-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output-dir', type=click.Path())
@click.argument('landmarks-path', type=click.Path())
@click.option('--batch-size', '-b', type=int, default=6, show_default=True)
@click.option('--num-workers', '-j', type=int, default=12, show_default=True)
@click.option('--resample/--no-resample', '-r', default=False, show_default=True)
def main(input_path, checkpoint_path, output_dir, landmarks_path, batch_size, num_workers, resample):
    import torch
    from tqdm import tqdm
    import torchio as tio
    import models
    import datasets

    fps = get_paths(input_path)
    subjects = [tio.Subject(image=tio.ScalarImage(fp)) for fp in fps]  # key must be 'image' as in get_test_transform
    transform = tio.Compose((
        tio.ToCanonical(),
        datasets.get_test_transform(landmarks_path),
    ))
    if resample:
        transform = tio.Compose((
            tio.Resample(),
            transform,
            # tio.CropOrPad((264, 268, 144)),  # ################################# for BITE?
        ))
    dataset = tio.SubjectsDataset(subjects, transform)
    checkpoint = torch.load(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.get_unet().to(device)
    model.load_state_dict(checkpoint['model'])
    output_dir = Path(output_dir)
    model.eval()
    torch.set_grad_enabled(False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    output_dir.mkdir(exist_ok=True, parents=True)
    for batch in tqdm(loader):
        inputs = batch['image'][tio.DATA].float().to(device)
        seg = model(inputs).softmax(dim=1)[:, 1:].cpu() > 0.5
        for tensor, affine, path in zip(seg, batch['image'][tio.AFFINE], batch['image'][tio.PATH]):
            image = tio.LabelMap(tensor=tensor, affine=affine.numpy())
            path = Path(path)
            out_path = output_dir / path.name.replace('.nii', '_seg_cnn.nii')
            image.save(out_path)
    return 0


def get_paths(path):
    import utils
    path = Path(path)
    if path.is_file():
        fps = [path]
    elif path.is_dir():
        fps = utils.sglob(path, '*.nii.gz')
    return fps


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
