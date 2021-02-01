import numpy as np


EPSILON = np.spacing(1)
SPATIAL_DIMENSIONS = 2, 3, 4


class TverskyLoss:
    def __init__(self, *, alpha, beta, epsilon=None):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = EPSILON if epsilon is None else epsilon

    def __call__(self, output, target):
        loss = get_tversky_loss(
            output,
            target,
            self.alpha,
            self.beta,
            epsilon=self.epsilon,
        )
        return loss


class DiceLoss(TverskyLoss):
    def __init__(self, epsilon=None):
        super().__init__(alpha=0.5, beta=0.5, epsilon=epsilon)


def get_confusion(output, target):
    if output.shape != target.shape:
        message = (
            f'Shape of output {output.shape} and target {target.shape} differ')
        raise ValueError(message)

    num_dimensions = output.ndim
    if num_dimensions == 3:  # 3D image, typically during testing
        kwargs = {}
    else:  # 5D tensor, typically during training
        is_torch_tensor = not isinstance(output, np.ndarray)
        key = 'dim' if is_torch_tensor else 'axis'
        kwargs = {key: SPATIAL_DIMENSIONS}

    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0

    tp = (p0 * g0).sum(**kwargs)
    fp = (p0 * g1).sum(**kwargs)
    fn = (p1 * g0).sum(**kwargs)
    return tp, fp, fn


def get_tversky_score(output, target, alpha, beta, epsilon=None):
    """
    https://arxiv.org/pdf/1706.05721.pdf
    """
    epsilon = EPSILON if epsilon is None else epsilon
    tp, fp, fn = get_confusion(output, target)

    numerator = tp + epsilon
    denominator = tp + alpha * fp + beta * fn + epsilon
    score = numerator / denominator
    return score


def get_tversky_loss(*args, **kwargs):
    losses = 1 - get_tversky_score(*args, **kwargs)
    return losses


def get_f_score(output, target, beta, epsilon=None):
    """
    https://en.wikipedia.org/wiki/F1_score#Definition
    """
    epsilon = EPSILON if epsilon is None else epsilon
    confusion = get_confusion(output, target)

    precision = get_precision(confusion)
    recall = get_recall(confusion)

    score = (1 + beta**2) * (precision * recall) / (precision + recall + epsilon)
    return score


def get_f_loss(*args, **kwargs):
    losses = 1 - get_f_score(*args, **kwargs)
    return losses


def get_f_score_alternative(output, target, beta, epsilon=None):
    """
    See https://brenocon.com/blog/2012/04/f-scores-dice-and-jaccard-set-similarity/
    """
    beta_tversky = 1 / (1 + beta**2)
    alpha_tversky = 1 - beta_tversky

    score = get_tversky_score(
        output, target, alpha_tversky, beta_tversky, epsilon=epsilon)
    return score


def get_dice_score(output, target):
    alpha = beta = 0.5
    return get_tversky_score(output, target, alpha, beta)


def get_dice_loss(output, target):
    losses = 1 - get_dice_score(output, target)
    return losses


def get_iou_score(output, target):
    alpha = beta = 1
    return get_tversky_score(output, target, alpha, beta)


def get_iou_loss(output, target):
    losses = 1 - get_iou_score(output, target)
    return losses


def get_precision_(output, target):
    confusion = get_confusion(output, target)
    return get_precision(confusion)


def get_recall_(output, target):
    confusion = get_confusion(output, target)
    return get_recall(confusion)


def get_precision(confusion, epsilon=None):
    epsilon = EPSILON if epsilon is None else epsilon
    tp, fp, _ = confusion
    precision = (tp + epsilon) / (tp + fp + epsilon)
    return precision


def get_recall(confusion, epsilon=None):
    epsilon = EPSILON if epsilon is None else epsilon
    tp, _, fn = confusion
    recall = (tp + epsilon) / (tp + fn + epsilon)
    return recall


def get_dice_from_precision_and_recall(precision, recall):
    return 2 / (1/precision + 1/recall)
