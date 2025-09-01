import medpy.metric.binary as metrics
from itertools import cycle
import numpy as np
import torch
from src.utils.data_transforms import find_onehot_dimension
from typing import Optional, Union, Callable

is_overlap = {
    'Dice': True
}

def multiclass_score(result, reference, metric, num_classes):
    scores = []
    
    for i in range(1, num_classes+1): 
        result_i, reference_i = (result == i).astype(int), (reference==i).astype(int)
        scores.append(metric(result_i, reference_i))
    
    return scores

def Hausdorff(result, reference):
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return np.inf
    return metrics.hd(result, reference)

def HD95(result, reference):
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return np.inf
    return metrics.hd95(result, reference)

def ASSD(result, reference):
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return np.inf
    return metrics.assd(result, reference)

def Dice(result, reference):
    return metrics.dc(result, reference)

def compute_scores(data, num_classes, metric=Dice):
    scores = []
    
    for sample in data:
        result, reference = sample['seg'], sample['GT']
        score = multiclass_score(result, reference, metric, num_classes)
        scores.append(score)
    
    return np.array(scores)

def compute_scores_by_name(data, num_classes, metric=Dice):
    scores = {}

    for sample in data:
        name = sample['seg_name']
        result, reference = sample['seg'], sample['GT']
        score = multiclass_score(result, reference, metric, num_classes)
        scores[name] = score
    
    return scores

def sample_N(scores, N, n_buckets=10):
    """
    Samples a specific total number of items, distributing samples across buckets.
    If a bucket cannot fulfill its quota, the remaining samples are distributed
    evenly among the other buckets that have capacity.
    
    Args:
        scores: Array of scores.
        N: Total number of samples to take.
        n_buckets: Number of buckets to create between 0 and 1 (default is 10).

    Returns:
        np.array: Indices of the sampled items.
    """
    bins = np.linspace(0, 1, n_buckets + 1)
    bucket_indices = np.digitize(scores, bins, right=False) - 1
    buckets = [np.where(bucket_indices == i)[0] for i in range(n_buckets)]

    actual_samples_per_bucket = [0] * n_buckets
    
    initial_target = [N // n_buckets] * n_buckets
    remainder_N = N % n_buckets
    for i in range(n_buckets - remainder_N, n_buckets):
        initial_target[i] += 1

    for i in range(n_buckets):
        actual_samples_per_bucket[i] = min(initial_target[i], len(buckets[i]))

    samples_to_redistribute = N - sum(actual_samples_per_bucket)

    if samples_to_redistribute > 0:
        eligible_bucket_indices_list = [
            i for i in range(n_buckets) 
            if len(buckets[i]) > actual_samples_per_bucket[i]
        ]
        
        if not eligible_bucket_indices_list:
            print(f"Warning: Could not fulfill N={N} samples. Only {sum(actual_samples_per_bucket)} sampled as no more items available for redistribution.")
        else:
            bucket_cycler = cycle(eligible_bucket_indices_list)
            while samples_to_redistribute > 0:
                current_bucket_idx = next(bucket_cycler)
                if actual_samples_per_bucket[current_bucket_idx] < len(buckets[current_bucket_idx]):
                    actual_samples_per_bucket[current_bucket_idx] += 1
                    samples_to_redistribute -= 1
                elif all(actual_samples_per_bucket[idx] >= len(buckets[idx]) for idx in eligible_bucket_indices_list):
                    # If all eligible buckets are now full, break the loop
                    print(f"Warning: Could not fulfill N={N} samples. Reached capacity of all eligible buckets. Still needed {samples_to_redistribute} samples.")
                    break
    
    final_sampled_indices = []
    for i in range(n_buckets):
        num_to_sample = actual_samples_per_bucket[i]
        if num_to_sample > 0:
            final_sampled_indices.extend(np.random.choice(buckets[i], size=num_to_sample, replace=False))

    return np.array(final_sampled_indices)


def sample_balanced(scores, n_buckets=10, min_val=0, max_val=1):
    """
    Sample the same number of items from each bucket for each class (if multiclass),
    and return the union of indices without duplicates.

    Args:
        scores (np.array): Array of scores. Shape (N,) for single class, (N, C) for multiclass.
        n_buckets (int): Number of buckets to divide scores into.

    Returns:
        np.array: Unique indices of the sampled items.
    """
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]  # Convert to (N, 1) for unified handling
        
    n_classes = scores.shape[1]
    all_indices = None
    all_indices = []

    for c in range(n_classes):
        class_scores = scores[:, c]

        # Create bins and assign scores to buckets
        bins = np.linspace(min_val, max_val, n_buckets + 1)
        bucket_indices = np.digitize(class_scores, bins, right=False) - 1
        buckets = [np.where(bucket_indices == i)[0] for i in range(n_buckets)]
        min_bucket_size = min(len(b) for b in buckets if len(b) > 0)

        for b in buckets:
            if len(b) >= min_bucket_size:
                all_indices.extend(np.random.choice(b, size=min_bucket_size, replace=False))

    return np.array(sorted(set(all_indices)))


def per_label_dice(
    input: torch.Tensor,
    target: torch.Tensor,
    axes: Optional[Union[list, tuple]] = None,
    input_is_batch: bool = False,
    eps: float = 1e-8,
):
    assert input.shape == target.shape, "Input and target must be the same shape."
    assert find_onehot_dimension(input) is not None
    assert find_onehot_dimension(target) is not None

    if axes is None:
        if input_is_batch:
            spatial_dims = list(range(2, input.dim()))
        else:
            spatial_dims = list(range(1, input.dim()))

    intersection = torch.sum(input * target, dim=spatial_dims)  # intersection
    size_input = torch.sum(input, dim=spatial_dims)
    size_target = torch.sum(target, dim=spatial_dims)

    dice = 2 * intersection / (size_input + size_target + eps)
    dice[(intersection == 0) & ((size_input + size_target) == 0)] = 1

    if input_is_batch:
        return dice.mean(dim=0)
    return dice


def pairwise_jaccard_distance(
    A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    assert find_onehot_dimension(A) is not None
    assert find_onehot_dimension(B) is not None

    B = B.transpose(1, 5)
    intersection = torch.sum(A & B, dim=(3, 4))  # B x N x C x M
    union = torch.sum(A | B, dim=(3, 4))
    pairwise_jaccard_distances = 1 - (intersection / (union + eps))

    # Union 0 means both A and B are 0, which actually means a perfect prediction
    pairwise_jaccard_distances[(union == 0) & (intersection == 0)] = 1

    # Move channels to the beginning
    pairwise_jaccard_distances = pairwise_jaccard_distances.transpose(1, 2)

    return pairwise_jaccard_distances  # B x C x N x M


def generalised_energy_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: Callable = pairwise_jaccard_distance,
) -> torch.Tensor:
    """
    Calculate the (generalised) energy distance (https://en.wikipedia.org/wiki/Energy_distance)
    where x,y are torch.Tensors containing samples of the distributions to be
    compared for a given metric. It is used to measure diversity of segmentations.
    The input can either be an array of samples (N x C x Sx x Sy)
    or a batch of arrays of samples (B x N x C x Sx x Sy). In the former case the function returns
    a GED value per class (C). In the later case it returns a matrix of GED values of shape B x C.
        Parameters:
            x (torch.Tensor): One set of N samples
            y (torch.Tensor): Another set of M samples
            metric (function): a function implementing the desired metric
        Returns:
            The generalised energy distance of the two samples (float)
    """

    assert x.dim() == y.dim()
    if x.dim() == 4:
        input_is_batch = False
    elif x.dim() == 5:
        input_is_batch = True
    else:
        raise ValueError(
            f"Unexpected dimension of input tensors: {x.dim()}. Expected 4 or 5."
        )

    assert find_onehot_dimension(x) is not None
    assert find_onehot_dimension(y) is not None

    if not input_is_batch:
        x, y = x.unsqueeze(0), y.unsqueeze(0)

    def expectation_of_difference(a, b):
        N, M = a.shape[1], b.shape[1]
        A = torch.tile(
            a[:, :, :, :, :, None], (1, 1, 1, 1, 1, M)
        )  # B x N x C x Sx x Sy x M
        B = torch.tile(
            b[:, :, :, :, :, None], (1, 1, 1, 1, 1, N)
        )  # B x M x C x Sx x Sy x N
        return metric(A, B).mean(dim=(2, 3))

    Exy = expectation_of_difference(x, y)
    Exx = expectation_of_difference(x, x)
    Eyy = expectation_of_difference(y, y)

    ed = torch.sqrt(2 * Exy - Exx - Eyy)

    if not input_is_batch:
        ed = ed[0]

    return ed


def predict_dice(probs: torch.Tensor) -> torch.Tensor:
    """
    Computes soft DSC for all classes using the mean prediction as pseudo ground truth
    Parameters:
    - probs: Tensor of shape [C, H, W] — softmax probabilities
    Returns:
    - Tensor of shape [C-1] — soft DSC per class excluding background (class 0)
    """
    C, H, W = probs.shape
    
    # Get hard prediction as pseudo ground truth
    pred_hard = probs.argmax(dim=0)  # [H, W]
    dices = torch.zeros(C - 1, device=probs.device)

    for j in range(1, C):  # skip class 0 (background)
        p_j = probs[j]  # Soft probabilities for class j 
        gt_j = (pred_hard == j).float()  # Hard mask for class j (pseudo ground truth)
        
        # Compute soft Dice coefficient
        intersection = (p_j * gt_j).sum()
        union = p_j.sum() + gt_j.sum()
        dice = (2 * intersection) / union
        
        dices[j - 1] = dice

    return dices
