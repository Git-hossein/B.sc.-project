import numpy as np
from .inference import load_all_normed_embeddings, infer_similar_audio_ultra_fast_ultimate
from path_settings import paths_config
from typing import Dict
import pprint
import os
from typing import Dict, List, Optional, Union
import numpy as np


def evaluate_retrieval_metrics(
    raw_results: Dict[str, Union[np.ndarray, List[str]]],
    recall_k_vals: List[int] = [1, 5, 10, 50],
    ground_truth_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Computes standard retrieval metrics (Recall@K, Median Rank, Mean Rank, MRR)
    directly from the raw retrieval output.

    :param raw_results: Dictionary containing:
        - "queries": np.ndarray of shape (Q,)
        - "retrieved_names": np.ndarray of shape (N,)
        - "retrieved_indices": np.ndarray of shape (Q, K)
    :param recall_k_vals: List of K values for Recall@K evaluation.
    :param ground_truth_map: Optional mapping {query_name: ground_truth_audio_name_or_list}.
                             If None, assumes exact stem matching between query and audio.
    :return: Dictionary containing R@K, MedR, MeanRank, and MRR.
    """
    queries = raw_results["queries"]
    retrieved_names = raw_results["retrieved_names"]
    retrieved_indices = raw_results["retrieved_indices"]

    num_queries, total_k = retrieved_indices.shape

    # 1. Map audio filenames to their matrix column indices for O(1) lookup
    name_to_idx = {name: idx for idx, name in enumerate(retrieved_names)}
    stem_to_idx = {os.path.splitext(name)[0]: idx for idx, name in enumerate(retrieved_names)}

    # 2. Extract 1-based ranks of the true target for each query
    ranks = []

    for i, q in enumerate(queries):
        q_stem = os.path.splitext(q)[0]

        # Determine single target audio index
        if ground_truth_map is not None:
            target = ground_truth_map.get(q, ground_truth_map.get(q_stem))
            if target is None:
                continue
            target_stem = os.path.splitext(target)[0]
            target_idx = stem_to_idx.get(target_stem)
        else:
            # Default: query video stem == ground truth audio stem
            target_idx = stem_to_idx.get(q_stem)

        if target_idx is None:
            raise ValueError(
                f"Target audio for query '{q}' (expected stem: '{q_stem}') "
                f"was not found in the retrieved audio pool (`retrieved_names`)."
            )

        # Check where target appears in top-K retrieved indices
        matched_positions = np.where(retrieved_indices[i] == target_idx)[0]

        if len(matched_positions) > 0:
            # 1-based rank of the best-ranked positive candidate
            ranks.append(matched_positions[0] + 1)
        else:
            # Target fell outside the retrieved top-K
            ranks.append(np.inf)

    ranks = np.array(ranks)
    total_evaluated = len(ranks)

    if total_evaluated == 0:
        return {}

    # 3. Compute Metrics
    metrics = {}

    for k in recall_k_vals:
        recall = float(np.mean(ranks <= k) * 100.0)
        metrics[f"R@{k}"] = round(recall, 4)

    # Median Rank & Mean Rank (calculated over valid ranks)
    finite_ranks = ranks[np.isfinite(ranks)]
    median_rank = float(np.median(ranks)) if len(finite_ranks) == total_evaluated else float(np.median(ranks))
    metrics["MedR"] = round(median_rank, 4)
    mean_rank = float(np.mean(finite_ranks)) if len(finite_ranks) > 0 else np.nan
    metrics["MeanRank"] = round(mean_rank, 4)

    # Mean Reciprocal Rank: 1 / rank if within top-K, else 0.0
    reciprocal_ranks = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
    mean_reciprocal_ranks = float(np.mean(reciprocal_ranks))
    metrics["MRR"] = round(mean_reciprocal_ranks , 4)

    return metrics

if __name__ == "__main__":
    
    import os
    VAL_VID_DIR = ...
    VAL_AUDIO_DIR = ...
    TRAIN_AUDIO_DIR = paths_config.training.cav_mae_audio
    TRAIN_VID_DIR = paths_config.training.cav_mae_video_clip
    # inf_lst = infer_similar_audio_ultra_fast_ultimate(os.listdir(TRAIN_VID_DIR), TRAIN_VID_DIR, load_all_normed_embeddings(TRAIN_AUDIO_DIR), 1000, temp=0.01)
    inf_lst_raw = infer_similar_audio_ultra_fast_ultimate(os.listdir(TRAIN_VID_DIR), TRAIN_VID_DIR, load_all_normed_embeddings(TRAIN_AUDIO_DIR), 1000, temp=0.01, return_raw=True)
    print('done')
    # pprint.pp(evaluate_audio_rag_NEW(inf_lst, "train", retrieved_embs= retrieved_embs, gt_embs= gt_embs), sort_dicts=False)
    pprint.pp(evaluate_retrieval_metrics(inf_lst_raw), sort_dicts=False)
    