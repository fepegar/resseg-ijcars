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
@click.option('--batch-size', '-b', type=int, default=6, show_default=True)
@click.option('--num-workers', '-j', type=int, default=12, show_default=True)
@click.option('--gpu/--cpu', default=True, show_default=True)
@click.option('--threshold/--no-threshold', default=False, show_default=True)
@click.option('--augmentation/--no-augmentation', default=True, show_default=True)  # whether to use same augmentation as the one during training
@click.option('--save-volumes/--no-save-volumes', '-v', default=True, show_default=True)
@click.option('--interpolation', default='bspline', type=click.Choice(['linear', 'bspline']), show_default=True)
@click.option('--std-noise', default=0, type=float)
def main(
        input_path,
        checkpoint_path,
        output_dir,
        landmarks_path,
        num_iterations,
        csv_path,
        batch_size,
        num_workers,
        gpu,
        threshold,
        augmentation,
        save_volumes,
        interpolation,
        std_noise,
        ):
    import torch
    import pandas as pd
    import numpy as np
    import torchio as tio
    from tqdm import tqdm

    import models

    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.get_unet().to(device)
    model.load_state_dict(checkpoint['model'])
    output_dir = Path(output_dir)
    model.eval()
    torch.set_grad_enabled(False)

    fps = get_paths(input_path)
    mean_dir = output_dir / 'mean'
    std_dir = output_dir / 'std'
    # entropy_dir = output_dir / 'entropy'
    mean_dir.mkdir(parents=True, exist_ok=True)
    std_dir.mkdir(parents=True, exist_ok=True)
    # entropy_dir.mkdir(parents=True, exist_ok=True)

    records = []
    progress = tqdm(fps, unit='subject')
    for fp in progress:
        subject_id = fp.name[:4]
        progress.set_description(subject_id)
        image = tio.ScalarImage(fp)
        subject = tio.Subject(image=image)  # key must be 'image' as in get_test_transform
        transform = get_transform(augmentation, landmarks_path)
        dataset = tio.SubjectsDataset(num_iterations * [subject], transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )
        all_results = []
        for subjects_list_batch in tqdm(loader, leave=False, unit='batch'):
            inputs = torch.stack([subject.image.data for subject in subjects_list_batch]).float().to(device)
            with torch.cuda.amp.autocast():
                segs = model(inputs).softmax(dim=1)[:, 1:].cpu()
            iterable = list(zip(subjects_list_batch, segs))
            for subject, seg in tqdm(iterable, leave=False, unit='subject'):
                subject.image.set_data(seg)
                inverse_transform = subject.get_inverse_transform(warn=False)
                inverse_transforms = inverse_transform.transforms
                first = inverse_transforms[0]
                if hasattr(first, 'image_interpolation') and first.image_interpolation != 'linear':
                    first.image_interpolation = 'linear'  # force interp to be lin so probs stay in [0,1]
                subject_back = inverse_transform(subject)
                result = subject_back.image.data
                assert np.count_nonzero(result.numpy() < 0) == 0, 'neg values found in result'
                if threshold:
                    result = (result >= 0.5).float()
                all_results.append(result)
        result = torch.stack(all_results)

        volumes = result.sum(dim=(-3, -2, -1)).numpy()
        mean_volumes = volumes.mean()
        std_volumes = volumes.std()
        volume_variation_coefficient = std_volumes / mean_volumes
        q1, q3 = np.percentile(volumes, (25, 75))
        quartile_coefficient_of_dispersion = (q3 - q1) / (q3 + q1)

        record = dict(
            Subject=subject_id,
            VolumeMean=mean_volumes,
            VolumeSTD=std_volumes,
            VVC=volume_variation_coefficient,
            Q1=q1,
            Q3=q3,
            QCD=quartile_coefficient_of_dispersion,
        )

        if save_volumes:
            for i, volume in enumerate(volumes):
                record[f'Volume_{i}'] = volume

        records.append(record)

        mean = result.mean(dim=0)
        std = result.std(dim=0)
        # entropy = utils.get_entropy(result)

        mean_image = tio.ScalarImage(tensor=mean, affine=image.affine)
        std_image = tio.ScalarImage(tensor=std, affine=image.affine)
        # entropy_image = tio.ScalarImage(tensor=entropy, affine=image.affine)

        mean_path = mean_dir / fp.name.replace('.nii', '_mean.nii')
        std_path = std_dir / fp.name.replace('.nii', '_std.nii')
        # entropy_path = entropy_dir / fp.name.replace('.nii', '_entropy.nii')

        mean_image.save(mean_path)
        std_image.save(std_path)
        # entropy_image.save(entropy_path)

        # So it's updated during execution
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


def get_transform(augmentation, landmarks_path):
    import datasets
    import torchio as tio
    if augmentation:
        return datasets.get_train_transform(landmarks_path)
    else:
        preprocess = datasets.get_test_transform(landmarks_path)
        augment = tio.Compose((
            tio.RandomFlip(),
            tio.OneOf({
                tio.RandomAffine(): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            })
        ))
        return tio.Compose((preprocess, augment))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
