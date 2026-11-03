import numpy as np
import torch
from torch.nn import functional as F


def neighbor_weights(actions, observed_indices, target_indices=None):
    """Fixed action-distance rule; never consumes any unobserved response."""
    a = np.asarray(actions)
    target_indices = np.arange(len(a)) if target_indices is None else np.asarray(target_indices)
    if not len(observed_indices):
        return np.empty((len(target_indices), 0), int), np.empty((len(target_indices), 0))
    distance = ((a[target_indices, None] - a[np.asarray(observed_indices)][None])**2).sum(-1)
    order = np.argsort(distance, axis=1, kind='stable')[:, :3]
    selected = np.asarray(observed_indices)[order]
    weight = 1 / (np.take_along_axis(distance, order, axis=1) + 1e-4)
    return selected, weight / weight.sum(1, keepdims=True)


def residual_tensor(base, observed_prior, observed_true, observed_mask, weights):
    w = weights[..., None] * observed_mask
    residual = torch.where(observed_mask, observed_true-observed_prior, 0.)
    return base + (w*residual).sum(1)/w.sum(1).clamp(min=1e-12)


def correct_predictions(actions, prediction, observations):
    result = np.array(prediction, copy=True)
    if not observations:
        return result
    indices = [int(o[0]) for o in observations]
    neighbors, weights = neighbor_weights(actions, indices)
    lookup = {int(o[0]): o for o in observations}
    true = np.array([[lookup[int(i)][1] for i in row] for row in neighbors])
    mask = np.array([[lookup[int(i)][2] for i in row] for row in neighbors], bool)
    w = weights[..., None]*mask
    residual = np.where(mask, true-np.asarray(prediction)[neighbors, :60], 0.)
    result[:, :60] += (w*residual).sum(1)/np.maximum(w.sum(1), 1e-12)
    for i, y, m, _, _ in observations:
        result[i, :60] = np.where(m, y, result[i, :60])
    return result


def marginal_teacher(env):
    """Privileged TRAINING label, analytically identical to one true query reward."""
    descriptor, valid = env._d, env._valid
    available = env.available()
    values = np.zeros(len(available))
    pool, known = env._pool, [i for i in env.seen if valid[i]]
    if pool and known and env._base_distance > 1e-12:
        distance = np.linalg.norm(descriptor[pool, None]-descriptor[None], axis=-1)
        current = distance[:, known].min(1)
        values = np.maximum(current[:, None]-distance, 0.).mean(0)/env._base_distance
    values[~valid] = 0.
    if env._pairs:
        boundary = np.zeros(len(available))
        for a, b in env._pairs:
            if a in env.seen and b not in env.seen: boundary[b] += 1/len(env._pairs)
            if b in env.seen and a not in env.seen: boundary[a] += 1/len(env._pairs)
        values = (values+boundary)/2
    values[~available] = 0.
    return values, available & valid


def ranking_loss(logits, gains, eligible):
    difference = gains[:, :, None]-gains[:, None, :]
    weight = torch.where(eligible[:, :, None] & eligible[:, None, :] & (difference > 1e-4), difference, 0.)
    error = F.softplus(-(logits[:, :, None]-logits[:, None, :]))
    return (weight*error).sum()/weight.sum().clamp(min=1e-12)
