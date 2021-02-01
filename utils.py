from pathlib import Path

import torch
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
BOLD = '\033[1m'
ENDC = '\033[0m'


def print_color(string, color, bold=False):
    """
    Formats the string with colors for terminal prints
    """
    if bold is True:
        print(BOLD + color + string + ENDC)
    else:
        print(color + string + ENDC)


def sglob(path, pattern='*'):
    return list(sorted(path.glob(pattern)))


def get_stem(path):
    """
    '/home/user/image.nii.gz' -> 'image'
    """
    return Path(path).name.split('.')[0]


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_tuple(value, length: int = 1):
    try:
        iter(value)
        value = tuple(value)
    except TypeError:
        value = length * (value,)
    return value


def get_entropy(tensor, epsilon=1e-6):
    tensor = torch.stack((1 - tensor, tensor), dim=1)
    mean = tensor.mean(dim=0) + epsilon  # avoid NaNs in log
    h = - (mean * mean.log()).sum(dim=0)
    assert np.count_nonzero(np.isnan(h)) == 0
    return h


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_median_iqr(x):
    p25, p50, p75 = np.percentile(x, (25, 50, 75))
    return np.array((p50, p75 - p25))


def print_significance(
        name_1,
        name_2,
        dice_1,
        dice_2,
        num_experiments=None,
        alpha=0.05,
        ):
    U, p = mannwhitneyu(dice_1, dice_2, alternative='less')
    if num_experiments is None:
        bonferroni_factor = 1
    else:
        bonferroni_factor = (num_experiments * (num_experiments - 1)) / 2
    m, i = get_median_iqr(dice_1)
    print(f'Median (IQR) of {name_1}: {m:.1f} ({i:.1f})')
    m, i = get_median_iqr(dice_2)
    print(f'Median (IQR) of {name_2}: {m:.1f} ({i:.1f})')
    print(f'{name_2} better than {name_1}: ', end='')
    if p <= 0.0001 / bonferroni_factor:
        print_color('****', BLUE)
    elif p <= 0.001 / bonferroni_factor:
        print_color('***', CYAN)
    elif p <= 0.01 / bonferroni_factor:
        print_color('**', GREEN)
    elif p <= 0.05 / bonferroni_factor:
        print_color('*', YELLOW)
    else:
        print_color('ns', RED)
    # if p < alpha:
    #     print('Significant difference')
    # else:
    #     print('No significant difference')
    print('U:', U)
    print('p:', p)
    print()


def read_df(path):
    return pd.read_csv(path, index_col=0, dtype={'Subject': str})


def get_df(experiment):
    path = Path(__file__).parent / 'runs' / str(experiment) / 'evaluation.csv'
    return read_df(path)


def get_dices(experiment):
    return get_df(experiment).Dice.values


def merge_dfs_in_dir(directory):
    directory = Path(directory)
    dfs = [read_df(path) for path in directory.glob('*.csv')]
    # https://stackoverflow.com/a/46100235/3956024
    return pd.concat(dfs, ignore_index=True).sort_values(by='Subject')
