from path_settings import paths_config
# import the modules accordingly
# from video_processing import vggsound_extract_video_embeddings_batch, vggsound_batch_extract_frames_from_vids
# from audio_processing import vggsound_extract_audio_from_videos_batch, vggsound_extract_audio_embeddings_batch
# from vggsound_processing import check_integrity
import os
import csv
import numpy as np
import soundfile as sf
from PIL import Image
import csv
import pprint
from wakepy import keep 
import random
import json
import shutil
from typing import Optional, Any, Dict



# Legacy code: TODO either make function of out these or remove them
# # Create separate folders for full and trimmed clips
# full_videos_path = paths_config.training.full_videos
# trimmed_videos_path = paths_config.training.trimmed_videos
# video_frames_path = paths_config.training.frames
# audios_path = paths_config.training.audios
# audio_embeddings_path = paths_config.training.audio_embeddings
# video_embeddings_path = paths_config.training.video_embeddings
# inferred_example_path = paths_config.inferred_example_path
# os.makedirs(full_videos_path, exist_ok=True)
# os.makedirs(trimmed_videos_path, exist_ok=True)
# os.makedirs(audios_path, exist_ok=True)
# os.makedirs(audio_embeddings_path, exist_ok=True)
# os.makedirs(video_embeddings_path, exist_ok=True)
# os.makedirs(inferred_example_path, exist_ok=True)

# # Log files path
# download_and_trim_log_file = paths_config.training.down_trim_log
# extract_audio_log_file = paths_config.training.audio_log
# embeddings_audio_log_file = paths_config.training.audio_emb_log
# video_embeddings_log_file = paths_config.training.video_emb_log

# # Create log file with headers if it doesn't exist
# if not os.path.exists(download_and_trim_log_file):
#     with open(download_and_trim_log_file, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(["video_id", "download", "trim"])

# if not os.path.exists(extract_audio_log_file):
#     with open(extract_audio_log_file, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(["video_id", "extract_audio_status"])

# if not os.path.exists(embeddings_audio_log_file):
#     with open(embeddings_audio_log_file, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["video_id", "audio_embedding_status"])

# if not os.path.exists(video_embeddings_log_file):
#     with open(video_embeddings_log_file, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(["video_id", "embed_status"])






def load_all_normed_embeddings(embeddings_dir:str, names: Optional[list[str]] = None):
    """ Load and normalize embeddings from a directory; optionally only load embeddings for specified names """

    if names is not None:
        add_npy_extension = lambda f: f if f.endswith(".npy") else f"{os.path.splitext(f)[0]}.npy"
        files = [add_npy_extension(name) for name in names]

    else:
        files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith(".npy")])

    file_emb_dict = {}
    
    for f in files:
        full_path = os.path.join(embeddings_dir, f)

        if not os.path.exists(full_path):
            raise Exception(f"⚠️ cannot load {f}: Not found on disk.")
        
        emb = np.load(full_path).squeeze()
        # Pre-normalize for faster cosine similarity later
        norm = np.linalg.norm(emb)
        file_emb_dict[f] = emb / (norm if norm > 0 else 1e-9)
        
    return file_emb_dict





def find_top_k_similar(
        query_embs: np.ndarray, 
        all_emb_dict: Optional[dict[str, np.ndarray]]= None, 
        embeddings_dir: str = "path/to/embeddings", 
        top_k: int = 5, 
        T: float = 1.0):
    """Computes batched cosine similarities between query embeddings and an audio gallery.

    Returns the top-k matches along with temperature-scaled softmax scores.

    Args:
        query_embs: 2D array of query embeddings of shape (Q, D).
        all_emb_dict: Pre-loaded dictionary mapping audio filenames to normalized embeddings. 
            If None, embeddings are automatically loaded from `embeddings_dir`.
        embeddings_dir: Path to load audio embeddings from if `all_emb_dict` is None.
        top_k: Number of top similar audio documents to retrieve per query.
        T: Temperature scaling factor for the softmax probability computation.

    Returns:
        A raw results dictionary containing retrieved filenames, indices, cosine similarities, 
        and softmax scores as NumPy arrays.
    """

    if all_emb_dict is None:

        all_emb_dict = load_all_normed_embeddings(embeddings_dir)

    AUDIO_FILENAMES = list(all_emb_dict.keys())
    ALL_AUDIO_EMBS = np.array(list(all_emb_dict.values()))
    

    # If a single query is passed (1D list), convert it to (1, D) as we always need a 2Dmatrix
    if query_embs.ndim == 1:
        query_embs = query_embs[np.newaxis, :]

    # 1. Vectorized Normalization (per row/query)
    q_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_ready = query_embs / (q_norms + 1e-9)
    
    # 2. BATCH MATRIX OPERATION
    # (Q, D) @ (D, N) -> (Q, N) Matrix of all similarities
    all_similarities = query_ready @ ALL_AUDIO_EMBS.T
    
    # 3. Process results for each query
    num_queries = all_similarities.shape[0]
    num_docs = all_similarities.shape[1]
    top_k = min(top_k, num_docs)
  
    batch_results_raw = {'retrieved_names': np.array(AUDIO_FILENAMES),
                        'retrieved_indices': np.empty((num_queries, top_k), dtype=np.int32), 
                        'cosine_sim': np.empty((num_queries, top_k), dtype=np.float32), 
                        'softmax_score': np.empty((num_queries, top_k), dtype=np.float32)}


    for i in range(num_queries):
        similarities = all_similarities[i]
        
        # perform partial sort if top k isn't too big, otherwise just sort everything (simple speedup trick)
        if top_k >= num_docs * 0.2:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            idx = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = idx[np.argsort(similarities[idx])][::-1]
        
        top_scores = similarities[top_indices]
        
        
        # Softmax on top K
        scores_shifted = top_scores / T
        exps = np.exp(scores_shifted - np.max(scores_shifted))
        softmax_probs = (exps / np.sum(exps))

        batch_results_raw["retrieved_indices"][i] = top_indices
        batch_results_raw["cosine_sim"][i] = top_scores
        batch_results_raw["softmax_score"][i] = softmax_probs

    
    return batch_results_raw




def convert_raw_to_nested_inference_dict(
    raw_results: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Converts a raw array-based retrieval/rerank dictionary into the legacy
    nested dictionary format:
    {
        query_name: {
            audio_name: {
                metric_1: float,
                metric_2: float,
                ...
            }
        }
    }
    Dynamically includes all score matrices present in raw_results
    (e.g., 'cosine_sim', 'softmax_score', 're_cosine_sim', 're_softmax_score').

    :param raw_results: Dict containing:
        - 'queries': np.ndarray (Q,)
        - 'retrieved_names': np.ndarray (N,)
        - 'retrieved_indices': np.ndarray (Q, K)
        - Plus one or more (Q, K) score arrays
    :return: Nested dictionary mapping queries -> retrieved docs -> metrics
    """
    queries = raw_results["queries"]
    retrieved_names = raw_results["retrieved_names"]
    retrieved_indices = raw_results["retrieved_indices"]

    # Identify all (Q, K) numeric metric arrays dynamically
    excluded_keys = {"queries", "retrieved_names", "retrieved_indices"}
    metric_keys = [
        k
        for k, v in raw_results.items()
        if k not in excluded_keys and isinstance(v, np.ndarray) and v.ndim == 2
    ]

    nested_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    num_queries, k = retrieved_indices.shape

    for i in range(num_queries):
        q_name = str(queries[i])
        query_dict: Dict[str, Dict[str, float]] = {}

        for rank in range(k):
            doc_idx = retrieved_indices[i, rank]
            doc_name = str(retrieved_names[doc_idx])

            # Extract every available metric for this match
            query_dict[doc_name] = {
                metric: round(float(raw_results[metric][i, rank]), 5)
                for metric in metric_keys
            }

        nested_results[q_name] = query_dict

    return nested_results


def infer_similar_audio(
        query_lst: list[str] | None = None, 
        query_dir = paths_config.training.video_embeddings,
        all_emb_dict: dict[str, np.ndarray]=load_all_normed_embeddings(paths_config.training.audio_embeddings), 
        top_k: int = 5, 
        random_sample: bool = False, 
        random_sample_count: int = 1, 
        temp: float = 1.0,
        return_raw:bool=False
    ):
    """Performs audio retrieval inference for video queries.

    Supports either evaluating a specific list of video query paths/names or randomly 
    sampling video queries from a directory.

    Args:
        query_lst: List of video query filenames or paths to evaluate. Required if `random_sample` is False. The queries may or may not include the 
        '.npy' extention in their names
        query_dir: Directory path containing the pre-computed video embedding .npy files.
        all_emb_dict: Dictionary of pre-loaded and normalized audio gallery embeddings.
        top_k: Number of top similar audio tracks to retrieve per video query.
        random_sample: If True, bypasses `query_lst` and randomly samples queries from `query_dir`.
        random_sample_count: Number of random video queries to sample if `random_sample` is True.
        temp: Temperature scaling parameter passed to softmax score calculations.
        return_raw: If True, returns the internal array-based raw dictionary; 
            if False, converts it to a human-readable nested dictionary.

    Returns:
        Retrieval results either in raw array format or nested dictionary format.
    """

    video_embeddings_dir = query_dir
    
    # 1. ensure all_emb_dict is an actual dict
    if not isinstance(all_emb_dict, dict):
        raise Exception("Please make sure that the audio gallery is provided as normalized dictionary of embeddings")
        

    query_matrix = []
    final_keys = []

    # --- MODE A: SPECIFIC QUERIES(the queries are specified) ---
    if not random_sample:
        if not query_lst:
            raise ValueError("query_lst must be provided when random_sample is False.")
        
        for q in query_lst:
            # Handle full path vs ID
            path = q if os.path.exists(q) else os.path.join(video_embeddings_dir, f"{os.path.splitext(q)[0]}.npy")
            if not os.path.exists(path):
                print(f"Warning: Skipping {q}, file not found.")
                continue
            
            emb = np.load(path).squeeze()
            query_matrix.append(emb)
            final_keys.append(os.path.basename(path))

    # --- MODE B: RANDOM SAMPLES(random sample is turned on) ---
    else:
        all_video_files = [f for f in os.listdir(video_embeddings_dir) if f.endswith(".npy")]
        if not all_video_files:
            raise FileNotFoundError("No video embeddings found.")
        
        count = min(random_sample_count, len(all_video_files))
        final_keys = random.sample(all_video_files, count)
        
        for name in final_keys:
            emb = np.load(os.path.join(video_embeddings_dir, name)).squeeze()
            query_matrix.append(emb)

    if not query_matrix:
        return {}

    # 2. convert to numpy array (matrix)
    query_matrix = np.array(query_matrix)


    top_similars_batch = find_top_k_similar(
        query_matrix, 
        all_emb_dict, 
        top_k=top_k, 
        T=temp
    )
    top_similars_batch["queries"]=np.array(final_keys)
    return top_similars_batch if return_raw else convert_raw_to_nested_inference_dict(top_similars_batch)



def create_inference_structured_folder(inference_dict: dict, inference_dir: str, query_videos_dir: str, audio_matches_dir: str):
    """
    Given a dictionary returned by `infer_similar_audio` and destination directory, creates a folder structure
    with trimmed videos and matched audio files for easy viewing.

    Folder structure:
    inference_dir/
        query_video_name/
            query_video_name.mp4
            matched_audio1.wav
            matched_audio2.wav
            ...

    Args:
        inference_dict (dict): output of `infer_similar_audio`
        inference_dir (str): path to create inference examples in
        query_videos_dir (str): where to look for the actual mp4 files of the video queries
        audio_matches_dir (str): where to look for the audio matches found for each video query
    """
    for video_key, audio_matches in inference_dict.items():

        # Remove .npy from video key to get folder/video name
        video_name = os.path.splitext(video_key)[0]
        video_folder = os.path.join(inference_dir, video_name)

        # Create or replace folder
        if os.path.exists(video_folder):
            shutil.rmtree(video_folder)
        os.makedirs(video_folder, exist_ok=True)


        #create json file
        json_file = os.path.join(video_folder, "inference_results.json")
        with open(json_file, "w") as f:
            json.dump(audio_matches, f, indent=4)

        # Copy trimmed video
        trimmed_video_file = os.path.join(query_videos_dir, f"{video_name}.mp4")
        if os.path.exists(trimmed_video_file):
            shutil.copy(trimmed_video_file, os.path.join(video_folder, f"{video_name}.mp4"))
        else:
            print(f"⚠️ Trimmed video not found for {video_name}, skipping video copy.")

        # Copy matched audio files
        for index, audio_file in enumerate(audio_matches.keys(), start=1):
            base_audio_name = os.path.splitext(audio_file)[0]
            audio_source = os.path.join(audio_matches_dir, f"{base_audio_name}.wav")
            ranked_filename = f"{base_audio_name}_rank{index}.wav"
            dest_path = os.path.join(video_folder, ranked_filename)

            if os.path.exists(audio_source):
                shutil.copy(audio_source, dest_path)
            else:
                print(f"⚠️ Audio file {base_audio_name}.wav not found for {video_name}, skipping.")

    print(f"☑️ Inference examples created in {inference_dir}")



if __name__ == "__main__":
    pass