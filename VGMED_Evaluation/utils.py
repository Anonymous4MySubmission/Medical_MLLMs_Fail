import numpy as np


NUM_IMG_TOKENS = 576
PATCHES = 24
SIZE = (336, 336)


def attention_ratio_vectorized(visual_attentions, gt_tokens):  # for visual_attentions = (..., 576)

    # Convert gt_tokens to flat indices
    token_indices = np.array([token[1] * PATCHES + token[0] for token in gt_tokens])

    # Gather relevant attention values
    relevant_attention = np.sum(visual_attentions[..., token_indices], axis=-1)

    # Compute average attention
    average_attention = np.sum(visual_attentions, axis=-1) / NUM_IMG_TOKENS * len(token_indices)

    # Compute attention ratio
    return relevant_attention / (average_attention + 1e-8)


def attention_IoU_vectorized(att_maps, gt_tokens):  # for att_maps = (28*28*prompt*576)

    att_maps = att_maps / (att_maps.sum(axis=-1, keepdims=True) + 1e-8)

    gt_maps = np.zeros(att_maps.shape[-1])
    for token in gt_tokens:
        gt_maps[token[1] * PATCHES + token[0]] = 1
    gt_maps = gt_maps / gt_maps.sum()

    attention_I = np.sum(att_maps * gt_maps, axis=-1)
    avg_map = (att_maps + gt_maps) / 2
    attention_U = np.sum(np.square(avg_map), axis=-1)

    return attention_I / attention_U


def js_divergence_vectorized(att_map, gt_tokens, epsilon=1e-8):

    att_map = att_map / (att_map.sum(axis=-1, keepdims=True) + epsilon)

    gt_mask = np.zeros(att_map.shape[-1])
    for token in gt_tokens:
        gt_mask[token[1] * PATCHES + token[0]] = 1
    gt_mask = gt_mask / gt_mask.sum()
    
    m = 0.5 * (att_map + gt_mask) 
    kl_att_m = np.sum(att_map * np.log((att_map + epsilon) / (m + epsilon)), axis=-1)
    kl_gt_m = np.sum(gt_mask * np.log((gt_mask + epsilon) / (m + epsilon)), axis=-1)
    js_div = 0.5 * (kl_att_m + kl_gt_m)

    return js_div


def kl_divergence_vectorized(att_map, gt_tokens, epsilon=1e-8):

    att_map = att_map / (att_map.sum(axis=-1, keepdims=True) + epsilon)

    gt_mask = np.zeros(att_map.shape[-1])
    for token in gt_tokens:
        gt_mask[token[1] * PATCHES + token[0]] = 1
    gt_mask = gt_mask / gt_mask.sum()

    epsilon = 1e-12
    att_map = np.clip(att_map, epsilon, 1)
    gt_mask = np.clip(gt_mask, epsilon, 1)

    return np.sum(gt_mask * np.log(gt_mask / att_map), axis=-1)