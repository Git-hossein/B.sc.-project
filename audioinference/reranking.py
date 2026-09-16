import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import os
from .inference import convert_raw_to_nested_dict


def sync_rerank(
    retrieval_results: Dict[str, np.ndarray],
    vid_query_sync_dir: str,
    audio_sync_dir_or_dict: Union[str, Dict[str, np.ndarray]],
    T: float = 1.0,
    return_raw: bool = True,
    batch_size: int = 16,
) -> Union[Dict[str, np.ndarray], Dict[str, Dict[str, Dict[str, float]]]]:
    add_npy = lambda f: f if str(f).endswith(".npy") else f"{os.path.splitext(str(f))[0]}.npy"
    queries = retrieval_results["queries"]
    retrieved_names = retrieval_results["retrieved_names"]
    retrieved_indices = retrieval_results["retrieved_indices"]
    orig_cosine_sim = retrieval_results["cosine_sim"]
    orig_softmax_score = retrieval_results["softmax_score"]

    num_queries, k = retrieved_indices.shape
    num_audios = len(retrieved_names)

    # -------------------------------------------------------------
    # STEP 1: Pre-load candidate pool once (N files instead of Q * K)
    # -------------------------------------------------------------
    print(f"Pre-loading {num_audios} audio temporal embeddings once into RAM...")
    audio_is_dict = isinstance(audio_sync_dir_or_dict, dict)
    
    # Load first sample to infer (T, D)
    sample_name = retrieved_names[0]
    if audio_is_dict:
        sample_emb = audio_sync_dir_or_dict.get(sample_name, audio_sync_dir_or_dict[os.path.splitext(sample_name)[0]])
    else:
        sample_emb = np.load(os.path.join(audio_sync_dir_or_dict, add_npy(sample_name)))
    num_frames, feat_dim = np.squeeze(sample_emb).shape

    # Contiguous block: (N, T, D)
    all_audio_matrix = np.empty((num_audios, num_frames, feat_dim), dtype=np.float16)

    for idx, name in enumerate(retrieved_names):
        if audio_is_dict:
            emb = audio_sync_dir_or_dict.get(name, audio_sync_dir_or_dict[os.path.splitext(name)[0]])
        else:
            path = os.path.join(audio_sync_dir_or_dict, add_npy(name))
            emb = np.load(path)
        
        emb = np.squeeze(emb).astype(np.float32)
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        normed = emb / (norm + 1e-9)
        all_audio_matrix[idx] = normed.astype(np.float16)

    print("Audio embeddings loaded. Starting vectorized re-ranking...")

    # -------------------------------------------------------------
    # STEP 2: Pre-allocate outputs
    # -------------------------------------------------------------
    new_retrieved_indices = np.empty((num_queries, k), dtype=np.int32)
    new_cosine_sim = np.empty((num_queries, k), dtype=np.float32)
    new_softmax_score = np.empty((num_queries, k), dtype=np.float32)
    re_cosine_sim = np.empty((num_queries, k), dtype=np.float32)
    re_softmax_score = np.empty((num_queries, k), dtype=np.float32)

    # -------------------------------------------------------------
    # STEP 3: Batch process queries via RAM indexing
    # -------------------------------------------------------------
    for start_idx in range(0, num_queries, batch_size):
        end_idx = min(start_idx + batch_size, num_queries)
        b_len = end_idx - start_idx

        # Load batch of queries (only B disk reads)
        batch_q = np.empty((b_len, num_frames, feat_dim), dtype=np.float32)
        for i, q_idx in enumerate(range(start_idx, end_idx)):
            q_name = queries[q_idx]
            q_path = os.path.join(vid_query_sync_dir, add_npy(q_name))
            q_emb = np.squeeze(np.load(q_path)).astype(np.float32)
            q_norm = np.linalg.norm(q_emb, axis=-1, keepdims=True)
            batch_q[i] = q_emb / (q_norm + 1e-9)

        batch_cand_indices = retrieved_indices[start_idx:end_idx]  # Shape: (B, K)

        # In-memory slice: (B, K, T, D)
        batch_cands = all_audio_matrix[batch_cand_indices].astype(np.float32)

        # Vectorized dot product along diagonal
        batch_sim = np.einsum("bktd,btd->bk", batch_cands, batch_q) / float(num_frames)

        # Descending sort
        sort_order = np.argsort(-batch_sim, axis=-1)
        row_indices = np.arange(b_len)[:, np.newaxis]

        batch_sorted_sim = batch_sim[row_indices, sort_order].astype(np.float32)
        re_cosine_sim[start_idx:end_idx] = batch_sorted_sim
        new_retrieved_indices[start_idx:end_idx] = batch_cand_indices[row_indices, sort_order]
        new_cosine_sim[start_idx:end_idx] = orig_cosine_sim[start_idx:end_idx][row_indices, sort_order]
        new_softmax_score[start_idx:end_idx] = orig_softmax_score[start_idx:end_idx][row_indices, sort_order]

        # Guarded Softmax (Same as slicing function)
        scores_scaled = batch_sorted_sim / max(T, 1e-6)
        exps = np.exp(scores_scaled - np.max(scores_scaled, axis=-1, keepdims=True))
        denom = np.sum(exps, axis=-1, keepdims=True)
        re_softmax_score[start_idx:end_idx] = (exps / np.maximum(denom, 1e-12)).astype(np.float32)

    rerank_raw_results = {
        "queries": queries,
        "retrieved_names": retrieved_names,
        "retrieved_indices": new_retrieved_indices,
        "cosine_sim": new_cosine_sim,
        "softmax_score": new_softmax_score,
        "re_cosine_sim": re_cosine_sim,
        "re_softmax_score": re_softmax_score,
    }
    return rerank_raw_results if return_raw else convert_raw_to_nested_dict(rerank_raw_results)

def slice_down_retrieval_dict(
    raw_results: Dict[str, np.ndarray],
    k: int,
    T: float = 1.0,
    return_raw: bool = True,
) -> Union[Dict[str, np.ndarray], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Slices a raw retrieval/re-ranking dict down from current top-K to top-k.
    Preserves 1D metadata arrays, slices all 2D matrices, and then recomputes
    the softmax distributions across the remaining sliced candidates.

    :param raw_results: Raw dictionary containing (Q, K) matrices.
    :param k: New target number of top candidates (must be <= current K).
    :param T: Temperature scaling for softmax calculation.
    :param return_raw: If True, returns sliced raw dict; else converts to nested dict.
    :return: Sliced retrieval results in raw or nested format.
    """
    current_k = raw_results["retrieved_indices"].shape[1]
    target_k = min(k, current_k)

    # 1. Slice down all arrays first to minimize working memory
    sliced_results = {}
    for key, val in raw_results.items():
        if isinstance(val, np.ndarray) and val.ndim == 2:
            # Slices retrieved_indices, cosine_sim, softmax_score, re_cosine_sim, etc.
            sliced_results[key] = val[:, :target_k].copy()
        else:
            sliced_results[key] = val

    # Helper for in-place, memory-stable softmax recalculation over sliced (Q, target_k)
    def recompute_softmax(scores: np.ndarray) -> np.ndarray:
        if scores.shape[-1] == 1:
            return np.ones_like(scores, dtype=np.float32)
        scores_scaled = scores / max(T, 1e-6)
        shifted = scores_scaled - np.max(scores_scaled, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        denom = np.sum(exps, axis=-1, keepdims=True)
        return (exps / np.maximum(denom, 1e-12)).astype(np.float32)

    # 2. Recompute softmaxes on the freshly sliced (Q, target_k) arrays
    if "cosine_sim" in sliced_results:
        sliced_results["softmax_score"] = recompute_softmax(sliced_results["cosine_sim"])

    if "re_cosine_sim" in sliced_results:
        sliced_results["re_softmax_score"] = recompute_softmax(sliced_results["re_cosine_sim"])

    return sliced_results if return_raw else convert_raw_to_nested_dict(sliced_results)





def sync_rerank_and_slice(
        retrieval_results: Dict[str, np.ndarray], 
        vid_query_sync_dir: str,
        audio_sync_dir_or_dict: Union[str, Dict[str, np.ndarray]],
        new_k:int, 
        T:float, 
        return_raw:bool
)-> Dict[str, Dict[str, Dict[str, float]]]:

    reranked = sync_rerank(
        retrieval_results=retrieval_results, 
        vid_query_sync_dir=vid_query_sync_dir, 
        audio_sync_dir_or_dict=audio_sync_dir_or_dict, 
        T=T, 
        return_raw=True
    )
    return slice_down_retrieval_dict(
        raw_results=reranked, 
        k=new_k, 
        T=T, 
        return_raw=return_raw
    )





def evaluate_reranking(
    orig_raw_results: Dict[str, np.ndarray],
    reranked_raw_results: Dict[str, np.ndarray],
    ground_truth_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[int, float]]:
    """
    Evaluates top-1 rank dynamics between original retrieval and re-ranking.
    Reports promotions/recoveries (moved into rank 1) and demotions/evictions
    (knocked out of rank 1).

    :param orig_raw_results: Raw retrieval dictionary before re-ranking.
    :param reranked_raw_results: Raw retrieval dictionary after re-ranking.
    :param ground_truth_map: Optional mapping {query_name: gt_audio_name}.
                             If None, assumes query stem == audio stem.
    :return: Dictionary containing promoted, demoted, preserved, and net top-1 counts.
    """
    queries = orig_raw_results["queries"]
    retrieved_names = orig_raw_results["retrieved_names"]
    orig_top1 = orig_raw_results["retrieved_indices"][:, 0]
    rerank_top1 = reranked_raw_results["retrieved_indices"][:, 0]

    stem_to_idx = {os.path.splitext(name)[0]: idx for idx, name in enumerate(retrieved_names)}

    promoted = 0
    demoted = 0
    preserved = 0
    total_evaluated = 0

    for i, q in enumerate(queries):
        q_stem = os.path.splitext(q)[0]

        if ground_truth_map is not None:
            target_name = ground_truth_map.get(q, ground_truth_map.get(q_stem))
            if target_name is None:
                continue
            target_idx = stem_to_idx.get(os.path.splitext(target_name)[0])
        else:
            target_idx = stem_to_idx.get(q_stem)

        if target_idx is None:
            raise ValueError(f"Ground truth target could not be resolved for query '{q}' (idx: {target_idx}).")

        total_evaluated += 1

        orig_hit = (orig_top1[i] == target_idx)
        rerank_hit = (rerank_top1[i] == target_idx)

        if not orig_hit and rerank_hit:
            promoted += 1
        elif orig_hit and not rerank_hit:
            demoted += 1
        elif orig_hit and rerank_hit:
            preserved += 1

    net_gain = promoted - demoted
    pct_net_gain = (net_gain / total_evaluated) * 100.0 if total_evaluated > 0 else 0.0
    orig_hits = preserved + demoted
    eviction_rate = (demoted / orig_hits) * 100.0 if orig_hits > 0 else 0.0
    return {
        "total_evaluated": total_evaluated,
        "promoted_to_top1": promoted,
        "demoted_from_top1": demoted,
        "preserved_at_top1": preserved,
        "net_top1_gain": promoted - demoted,
        "net_top1_gain_pct %": round(pct_net_gain, 2),
        "eviction_rate_pct %": round(eviction_rate, 2)
    }




if __name__ == "__main__":
    from path_settings import paths_config
    from audioinference.inference import infer_similar_audio_ultra_fast_ultimate, load_all_normed_embeddings
    from audioinference.evaluation import evaluate_retrieval_metrics
    import pprint 

    print("RERANKING but with clip wav2clip")

    myinfer = infer_similar_audio_ultra_fast_ultimate(
        os.listdir(paths_config.training.video_embeddings)[:5000], 
        paths_config.training.video_embeddings,
        load_all_normed_embeddings(paths_config.training.audio_embeddings),
        50, 
        False, 
        5000, 
        0.01,
        return_raw=True
)

    # pprint.pprint(convert_raw_to_nested_dict(myinfer), sort_dicts=False)
    pprint.pp(evaluate_retrieval_metrics(myinfer))
    print("\n\n\n rerank:\n")

    renranked = sync_rerank(myinfer, paths_config.training.cav_mae_sync_video, paths_config.training.cav_mae_sync_audio, 0.01, return_raw=True)
    # pprint.pp(convert_raw_to_nested_dict(renranked), sort_dicts=False)


    print("\n\n\n eval:\n")


    pprint.pp(evaluate_reranking(myinfer, renranked))
    print("\n\n")
    pprint.pp(evaluate_retrieval_metrics(renranked))