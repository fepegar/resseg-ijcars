# pylint: skip-file
# Pylint said Maximum recursion depth exceeded

import sys
import click
from pathlib import Path


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
@click.argument('checkpoint-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output-dir', type=click.Path())
@click.argument('landmarks-path', type=click.Path())
@click.argument('num-iterations', type=int)
@click.argument('csv-path', type=click.Path())
@click.option('--num-workers', '-j', type=int, default=12, show_default=True)
@click.option('--gpu/--cpu', default=True, show_default=True)
@click.option('--threshold/--no-threshold', default=False, show_default=True)
@click.option('--interpolation', default='bspline', type=click.Choice(['linear', 'bspline']), show_default=True)
def main(
        input_path,
        checkpoint_path,
        output_dir,
        landmarks_path,
        num_iterations,
        csv_path,
        num_workers,
        gpu,
        threshold,
        interpolation,
        ):
    import torch
    import pandas as pd
    import numpy as np
    import torchio as tio
    from tqdm import tqdm, trange

    import utils
    import models
    import datasets

    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.get_unet().to(device)
    model.load_state_dict(checkpoint['model'])
    output_dir = Path(output_dir)
    model.eval()
    utils.enable_dropout(model)

    torch.set_grad_enabled(False)

    fps = get_paths(input_path)
    mean_dir = output_dir / 'mean'
    std_dir = output_dir / 'std'
    entropy_dir = output_dir / 'entropy'
    mean_dir.mkdir(parents=True, exist_ok=True)
    std_dir.mkdir(parents=True, exist_ok=True)
    entropy_dir.mkdir(parents=True, exist_ok=True)

    records = []
    progress = tqdm(fps, unit='subject')
    for fp in progress:
        subject_id = fp.name[:4]
        progress.set_description(subject_id)
        image = tio.ScalarImage(fp)
        subject = tio.Subject(image=image)  # key must be 'image' as in get_test_transform
        preprocess = datasets.get_test_transform(landmarks_path)
        preprocessed = preprocess(subject)
        inputs = preprocessed.image.data.float()[np.newaxis].to(device)
        all_results = []
        for _ in trange(num_iterations, leave=False):
            with torch.cuda.amp.autocast():
                segs = model(inputs).softmax(dim=1)[0, 1:]
            all_results.append(segs.cpu())
        result = torch.stack(all_results)

        volumes = result.sum(dim=(-3, -2, -1)).numpy()
        mean_volumes = volumes.mean()
        std_volumes = volumes.std()
        volume_variation_coefficient = std_volumes / mean_volumes
        q1, q3 = np.percentile(volumes, (25, 75))
        quartile_coefficient_of_dispersion = (q3 - q1) / (q3 + q1)

        records.append(
            dict(
                Subject=subject_id,
                VolumeMean=mean_volumes,
                VolumeSTD=std_volumes,
                VVC=volume_variation_coefficient,
                Q1=q1,
                Q3=q3,
                QCD=quartile_coefficient_of_dispersion,
            )
        )

        crop: tio.Crop = preprocessed.history[-1]
        pad = crop.inverse()

        assert np.count_nonzero(result.numpy() < 0) == 0, 'neg values found in result'
        mean = result.mean(dim=0)
        assert np.count_nonzero(mean.numpy() < 0) == 0, 'neg values found in mean'
        std = result.std(dim=0)
        # entropy = utils.get_entropy(result)

        mean_image = tio.ScalarImage(tensor=mean, affine=preprocessed.image.affine)
        std_image = tio.ScalarImage(tensor=std, affine=preprocessed.image.affine)
        # entropy_image = tio.ScalarImage(tensor=entropy, affine=preprocessed.image.affine)

        mean_path = mean_dir / fp.name.replace('.nii', '_mean.nii')
        std_path = std_dir / fp.name.replace('.nii', '_std.nii')
        # entropy_path = entropy_dir / fp.name.replace('.nii', '_entropy.nii')

        pad(mean_image).save(mean_path)
        pad(std_image).save(std_path)
        # pad(entropy_image).save(entropy_path)

        # So it's updated while it runs
        df = pd.DataFrame.from_records(records)
        df.to_csv(csv_path)

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
