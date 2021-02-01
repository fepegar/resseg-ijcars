import hashlib
from pathlib import Path

import torch
import pandas as pd
import torchio as tio
from tqdm import tqdm
from resector import RandomResection
from sklearn.model_selection import KFold

from utils import sglob, get_stem


class DataModule:
    def __init__(
            self,
            datasets_dir,
            train_batch_size,
            num_workers,
            ):
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.datasets_dir = Path(datasets_dir).expanduser()

    def get_train_loader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def get_val_loader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def get_train_transform(self, resect=True):
        return get_train_transform(self.landmarks_path, resection_params=self.resection_params)

    def print_lengths(self, test=True):
        f = print if self.log is None else self.log.info
        f(f'{len(self.train_dataset):4} training instances')
        f(f'{len(self.train_loader):4} training batches')
        f(f'{len(self.val_dataset):4} validation instances')
        f(f'{len(self.val_loader):4} validation batches')
        if not test:
            return
        f(f'{len(self.test_dataset):4} test instances')
        f(f'{len(self.test_loader):4} test batches')

    def get_public_subjects(self):
        public_dataset_names = (
            'IXI',
            'ADNI1_15T',
            'ADNI1_3T',
            'ADNI2',
            'OASIS_download',
        )
        all_subjects = []
        for name in public_dataset_names:
            subjects = get_subjects_list_from_dir(self.datasets_dir / name)
            all_subjects.extend(subjects)
        return all_subjects


class DataModulePublic(DataModule):
    def __init__(
            self,
            datasets_dir,
            real_dataset_dir,
            resection_params,
            train_batch_size,
            num_workers,
            pseudo_dir=None,
            split_ratio=0.9,
            split_seed=42,
            debug_ratio=0.02,
            log=None,
            debug=False,
            augment=True,
            verbose=False,
            cache_validation_set=True,
            histogram_standardization=True,
            ):
        super().__init__(datasets_dir, train_batch_size, num_workers)
        self.resection_params = resection_params

        # Precomputed from 90% of the public training data
        if histogram_standardization:
            self.landmarks_path = Path(__file__).parent / 'landmarks' / 'histogram_landmarks_default.npy'
        else:
            self.landmarks_path = None

        public_subjects = self.get_public_subjects()
        train_public, val_public = self.split_subjects(public_subjects, split_ratio, split_seed)

        train_transform = self.get_train_transform() if augment else self.get_val_transform()
        self.train_dataset = tio.SubjectsDataset(train_public, transform=train_transform)
        self.val_dataset = tio.SubjectsDataset(val_public, transform=train_transform)
        if cache_validation_set:
            self.val_dataset = cache(self.val_dataset, resection_params, augment=augment)
        test_transform = get_test_transform(self.landmarks_path)
        self.test_dataset = get_real_resection_dataset(real_dataset_dir, transform=test_transform)
        if debug:
            self.train_dataset = reduce_dataset(self.train_dataset, debug_ratio)
            self.val_dataset = reduce_dataset(self.val_dataset, debug_ratio)
            self.test_dataset = reduce_dataset(self.test_dataset, debug_ratio)

        self.train_loader = self.get_train_loader(self.train_dataset)
        self.val_loader = self.get_val_loader(self.val_dataset)
        self.test_loader = self.get_val_loader(self.test_dataset)

        self.log = log

        if verbose:
            self.print_lengths()

    @staticmethod
    def split_subjects(subjects, ratio, seed):
        len_subjects = len(subjects)
        len_training = int(len_subjects * ratio)
        len_validation = len_subjects - len_training
        lengths = len_training, len_validation
        with torch.random.fork_rng([]):
            torch.manual_seed(seed)
            train, val = torch.utils.data.random_split(subjects, lengths)
        return train, val

    def get_val_transform(self):
        return tio.Compose((get_simulation_transform(self.resection_params), get_test_transform(self.landmarks_path)))


class DataModuleCV(DataModule):
    def __init__(
            self,
            fold,
            num_folds,
            datasets_dir,
            dataset_name,
            train_batch_size,
            num_workers,
            use_public_landmarks=False,
            pseudo_dirname=None,
            split_seed=42,
            log=None,
            verbose=True,
            ):
        super().__init__(datasets_dir, train_batch_size, num_workers)
        self.resection_params = None
        real_dataset_dir = self.datasets_dir / 'real' / dataset_name
        real_subjects = get_real_resection_subjects(real_dataset_dir)
        train_subjects, val_subjects = self.split_subjects(real_subjects, fold, num_folds, split_seed)
        self.train_dataset = tio.SubjectsDataset(train_subjects)
        if use_public_landmarks:
            self.landmarks_path = get_landmarks_path()
        else:
            self.landmarks_path = get_landmarks_path(dataset=self.train_dataset)
        train_transform = self.get_train_transform(resect=False)
        self.train_dataset.set_transform(train_transform)
        test_transform = get_test_transform(self.landmarks_path)
        self.val_dataset = tio.SubjectsDataset(val_subjects, transform=test_transform)

        if pseudo_dirname is not None:
            pseudo_dir = self.datasets_dir / 'real' / pseudo_dirname
            pseudo_dataset = get_real_resection_dataset(pseudo_dir, transform=train_transform)
            self.train_dataset = torch.utils.data.ConcatDataset((self.train_dataset, pseudo_dataset))

        self.train_loader = self.get_train_loader(self.train_dataset)
        self.val_loader = self.test_loader = self.get_val_loader(self.val_dataset)

        self.log = log
        if verbose:
            self.print_lengths(test=False)

    @staticmethod
    def split_subjects(real_subjects, fold, num_folds, split_seed):
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=split_seed)
        folds = list(kf.split(real_subjects))
        train_indices, val_indices = folds[fold]
        train_subjects = [real_subjects[i] for i in train_indices]
        val_subjects = [real_subjects[i] for i in val_indices]
        return train_subjects, val_subjects


def get_train_transform(landmarks_path, resection_params=None):
    spatial_transform = tio.Compose((
        tio.OneOf({
            tio.RandomAffine(): 0.9,
            tio.RandomElasticDeformation(): 0.1,
        }),
        tio.RandomFlip(),
    ))
    resolution_transform = tio.OneOf((
            tio.RandomAnisotropy(),
            tio.RandomBlur(),
        ),
        p=0.75,
    )
    transforms = []
    if resection_params is not None:
        transforms.append(get_simulation_transform(resection_params))
    if landmarks_path is not None:
        transforms.append(tio.HistogramStandardization({'image': landmarks_path}))
    transforms.extend([
        # tio.RandomGamma(p=0.2),
        resolution_transform,
        tio.RandomGhosting(p=0.2),
        tio.RandomSpike(p=0.2),
        tio.RandomMotion(p=0.2),
        tio.RandomBiasField(p=0.5),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.RandomNoise(p=0.75),  # always after ZNorm and after blur!
        spatial_transform,
        get_tight_crop(),
    ])
    return tio.Compose(transforms)


def get_subjects_list_from_dir(dataset_dir):
    dataset_dir = Path(dataset_dir)
    mni_dir = dataset_dir / 'mni'
    resection_dir = dataset_dir / 'resection'
    noise_paths = sglob(resection_dir, '*noise*')
    subjects_list = []
    for noise_path in noise_paths:
        stem = noise_path.stem.split('_noise')[0]
        image_path = mni_dir / f'{stem}_on_mni.nii.gz'
        gml_path = resection_dir / f'{stem}_gray_matter_left_seg.nii.gz'
        gmr_path = resection_dir / f'{stem}_gray_matter_right_seg.nii.gz'
        rl_path = resection_dir / f'{stem}_resectable_left_seg.nii.gz'
        rr_path = resection_dir / f'{stem}_resectable_right_seg.nii.gz'
        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            resection_noise=tio.ScalarImage(noise_path),
            resection_gray_matter_left=tio.LabelMap(gml_path),
            resection_gray_matter_right=tio.LabelMap(gmr_path),
            resection_resectable_left=tio.LabelMap(rl_path),
            resection_resectable_right=tio.LabelMap(rr_path),
        )
        subjects_list.append(subject)
    return subjects_list


def get_landmarks_path(dataset=None):
    landmarks_dir = Path(__file__).parent / 'landmarks'
    landmarks_dir.mkdir(exist_ok=True)
    if dataset is None:  # get precomputed landmarks from public data
        landmarks_path = landmarks_dir / 'histogram_landmarks_default.npy'
    else:
        filename = f'histogram_landmarks_{get_stems_hash(dataset)}.npy'
        landmarks_path = landmarks_dir / filename
        if not landmarks_path.is_file():
            from torchio.transforms import train_histogram
            images_paths = [subject.image.path for subject in dataset.subjects]
            print('Training histogram landmarks:', landmarks_path)
            train_histogram(images_paths, output_path=landmarks_path)
    return landmarks_path


def get_stems_hash(dataset):
    # https://stackoverflow.com/a/27522708/3956024
    stems_string = ','.join(get_stem(subject.image.path) for subject in dataset.subjects)
    return hashlib.md5(stems_string.encode()).hexdigest()


def get_tight_crop():
    # Crop from (193, 229, 193) to (176, 216, 160)
    crop = tio.Crop((9, 8, 7, 6, 17, 16))
    return crop


def get_real_resection_subjects(dataset_dir):
    dataset_dir = Path(dataset_dir)
    image_dir = dataset_dir / 'image'
    label_dir = dataset_dir / 'label'
    image_paths = sglob(image_dir)
    label_paths = sglob(label_dir)
    assert len(image_paths) == len(label_paths)
    subjects = []
    for image_path, label_path in zip(image_paths, label_paths):
        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    return subjects


def get_real_resection_dataset(dataset_dir, transform=None):
    subjects = get_real_resection_subjects(dataset_dir)
    return tio.SubjectsDataset(subjects, transform=transform)


def reduce_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    return torch.utils.data.Subset(dataset, list(range(n)))


def cache(dataset, resection_params, augment=True, caches_dir='/tmp/val_set_cache', num_workers=12):
    caches_dir = Path(caches_dir)
    wm_lesion_p = resection_params['wm_lesion_p']
    clot_p = resection_params['clot_p']
    shape = resection_params['shape']
    texture = resection_params['texture']
    augment_string = '_no_augmentation' if not augment else ''
    dir_name = f'wm_{wm_lesion_p}_clot_{clot_p}_{shape}_{texture}{augment_string}'
    cache_dir = caches_dir / dir_name
    image_dir = cache_dir / 'image'
    label_dir = cache_dir / 'label'
    if not cache_dir.is_dir():
        print('Caching validation set')
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=lambda x: x[0],
        )
        for subject in tqdm(loader):
            image_path = image_dir / subject.image.path.name
            label_path = label_dir / subject.image.path.name  # label has no path because it was created not loaded
            subject.image.save(image_path)
            subject.label.save(label_path)

    subjects = []
    for im_path, label_path in zip(sglob(image_dir), sglob(label_dir)):
        subject = tio.Subject(
            image=tio.ScalarImage(im_path),
            label=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    return tio.SubjectsDataset(subjects)


def get_test_transform(landmarks_path):
    transforms = []
    if landmarks_path is not None:
        transforms.append(tio.HistogramStandardization({'image': landmarks_path}))
    transforms.extend([
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        get_tight_crop(),
    ])
    return tio.Compose(transforms)


def get_simulation_transform(resection_params):
    transform = RandomResection(
        volumes_range=(844, 83757),  # percentiles 1 and 99 of volumes in labeled EPISURG
        wm_lesion_p=resection_params['wm_lesion_p'],
        clot_p=resection_params['clot_p'],
        shape=resection_params['shape'],
        texture=resection_params['texture'],
    )
    return transform


def get_pseudo_loader(
        threshold,
        percentile,
        metric,
        summary_path,
        dataset_name,
        num_workers,
        batch_size=2,
        remove_zero_volume=False,
        ):
    subjects = []
    subject_ids = get_certain_subjects(
        threshold,
        percentile,
        metric,
        summary_path,
        remove_zero_volume=remove_zero_volume,
    )
    dataset_dir = Path('/home/fernando/datasets/real/') / dataset_name
    assert dataset_dir.is_dir()
    image_dir = dataset_dir / 'image'
    label_dir = dataset_dir / 'label'
    for subject_id in subject_ids:
        image_path = list(image_dir.glob(f'{subject_id}_*'))[0]
        label_path = list(label_dir.glob(f'{subject_id}_*'))[0]
        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    transform = get_train_transform(get_landmarks_path())
    dataset = tio.SubjectsDataset(subjects, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=num_workers,
    )
    return loader


def get_certain_subjects(
        threshold,
        percentile,
        metric,
        summary_path,
        remove_zero_volume=False,
        ):
    df = pd.read_csv(summary_path, index_col=0, dtype={'Subject': str})
    if remove_zero_volume:
        df = df[df.Volume > 0]
    column = df[metric]
    assert not (threshold is None and percentile is None)
    assert not (threshold is not None and percentile is not None)
    if percentile is not None:
        df = df[column < column.quantile(percentile / 100)]
    elif threshold is not None:
        df = df[column < threshold]
    return df.Subject.values
