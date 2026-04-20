from path_settings import paths_config
# import the modules accordingly
from video_processing import vggsound_extract_video_embeddings_batch, vggsound_batch_extract_frames_from_vids
from audio_processing import vggsound_extract_audio_from_videos_batch, vggsound_extract_audio_embeddings_batch
from vggsound_processing import check_integrity
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
from typing import Optional




# Create separate folders for full and trimmed clips
full_videos_path = paths_config.full_videos_path
trimmed_videos_path = paths_config.trimmed_videos_path
video_frames_path = paths_config.video_frames_path
audios_path = paths_config.audios_path
audio_embeddings_path = paths_config.audio_embeddings_path
video_embeddings_path = paths_config.video_embeddings_path
inferred_example_path = paths_config.inferred_example_path
os.makedirs(full_videos_path, exist_ok=True)
os.makedirs(trimmed_videos_path, exist_ok=True)
os.makedirs(audios_path, exist_ok=True)
os.makedirs(audio_embeddings_path, exist_ok=True)
os.makedirs(video_embeddings_path, exist_ok=True)
os.makedirs(inferred_example_path, exist_ok=True)

# Log files path
download_and_trim_log_file = paths_config.download_and_trim_log_file
extract_audio_log_file = paths_config.extract_audio_log_file
embeddings_audio_log_file = paths_config.embeddings_audio_log_file
video_embeddings_log_file = paths_config.video_embeddings_log_file

# Create log file with headers if it doesn't exist
if not os.path.exists(download_and_trim_log_file):
    with open(download_and_trim_log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "download", "trim"])

if not os.path.exists(extract_audio_log_file):
    with open(extract_audio_log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "extract_audio_status"])

if not os.path.exists(embeddings_audio_log_file):
    with open(embeddings_audio_log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "audio_embedding_status"])

if not os.path.exists(video_embeddings_log_file):
    with open(video_embeddings_log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "embed_status"])



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D numpy arrays.
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))

def softmax(scores, T = 1.0):
    scores = np.array(scores) / T
    scores_max = np.max(scores)
    exponentials = np.exp(scores - scores_max)
    return (exponentials / np.sum(exponentials)).tolist()




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






def find_top_k_similar_ultra_fast_ultimate(
        query_embs: np.ndarray, 
        all_emb_dict: Optional[dict[str, np.ndarray]]= None, 
        embeddings_dir: str = "path/to/embeddings", 
        k: int = 5, 
        T: float = 1.0):
    """
    Docstring for find_top_k_similar_ultra_fast_ultimate
    
    :param query_embs: Description
    :type query_embs: np.ndarray
    :param all_emb_dict: Description
    :type all_emb_dict: Optional[dict[str, np.ndarray]]
    :param embeddings_dir: Description
    :type embeddings_dir: str
    :param k: Description
    :type k: int
    :param T: Description
    :type T: float
    :return: Description
    :rtype: Any
    """

    # 1. Fallback Loading Logic
    if all_emb_dict is None:

        all_emb_dict = load_all_normed_embeddings(embeddings_dir)

    AUDIO_FILENAMES = list(all_emb_dict.keys())
    ALL_AUDIO_EMBS = np.array(list(all_emb_dict.values()))
    

    # 2. Ensure query is 2D (Matrix)
    # If a single query is passed (1D), convert it to (1, D)
    if query_embs.ndim == 1:
        query_embs = query_embs[np.newaxis, :]

    # 3. Vectorized Normalization (per row)
    q_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_ready = query_embs / (q_norms + 1e-9)
    
    # 4. BATCH MATRIX OPERATION
    # (Q, D) @ (D, N) -> (Q, N) Matrix of all similarities
    all_similarities = query_ready @ ALL_AUDIO_EMBS.T
    
    # 5. Process results for each query
    batch_results = []
    num_queries = all_similarities.shape[0]
    num_docs = all_similarities.shape[1]
    
    # Use a loop for the Top-K/Softmax part (this is O(Q), which is very fast)
    # The heavy lifting O(Q*N) was already done in the matrix multiplication above.
    for i in range(num_queries):
        similarities = all_similarities[i]
        
        # Top K Logic
        current_k = min(k, num_docs)
        if current_k >= num_docs * 0.2:
            top_indices = np.argsort(similarities)[::-1][:current_k]
        else:
            idx = np.argpartition(similarities, -current_k)[-current_k:]
            top_indices = idx[np.argsort(similarities[idx])][::-1]
        
        top_scores = similarities[top_indices]
        top_files = [AUDIO_FILENAMES[j] for j in top_indices]
        
        # Softmax on top K
        scores_shifted = top_scores / T
        exps = np.exp(scores_shifted - np.max(scores_shifted))
        softmax_probs = (exps / np.sum(exps)).tolist()
        
        query_results = []

        for filename, prob, cosine in zip(top_files, softmax_probs, top_scores):
            query_results.append({
                "filename": filename,
                "softmax_score": float(prob),
                "cosine_sim": float(cosine)})

        batch_results.append(query_results)
    
    # Return a single list if only one query was provided, else the whole batch
    return batch_results



def infer_similar_audio_ultra_fast_ultimate(
        query_lst: list[str] | None = None, 
        query_dir = paths_config.video_embeddings_path,
        all_emb_dict: Optional[dict[str, np.ndarray]]= None, 
        top_k: int = 5, 
        random_sample: bool = False, 
        random_sample_count: int = 1, 
        temp: float = 1.0
    ):
    """
    Docstring for infer_similar_audio_ultra_fast_ultimate
    
    :param query_lst: Description
    :type query_lst: list[str] | None
    :param all_emb_dict: Description
    :type all_emb_dict: Optional[dict[str, np.ndarray]]
    :param top_k: Description
    :type top_k: int
    :param random_sample: Description
    :type random_sample: bool
    :param random_sample_count: Description
    :type random_sample_count: int
    :param temp: Description
    :type temp: float
    """

    video_embeddings_dir = query_dir
    
    # 1. Ensure we have the audio embeddings loaded
    if all_emb_dict is None:
        print("Embeddings is None. Loading audio embeddings manually...")
        all_emb_dict = load_all_normed_embeddings(audio_embeddings_path)

    query_matrix = []
    final_keys = []

    # --- MODE A: SPECIFIC QUERIES ---
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

    # --- MODE B: RANDOM SAMPLES ---
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

    # 2. Convert to 2D Matrix (N, Dimension)
    query_matrix = np.array(query_matrix)

    # 3. Use the Batch Search Function
    # NOTE: Ensure your find_top_k_similar_fast handles 2D input as discussed
    top_similars_batch = find_top_k_similar_ultra_fast_ultimate(
        query_matrix, 
        all_emb_dict, 
        k=top_k, 
        T=temp
    )

    # 5. Build Result Dictionary
    results = {
        query: {similar["filename"]: {"softmax_score": similar["softmax_score"], "cosine_sim": similar["cosine_sim"]} for similar in similars}
        for query, similars in zip(final_keys, top_similars_batch)
    }

    return results





def create_inference_example_ultimate(inference_dict, inference_dir = inferred_example_path, query_videos_dir = trimmed_videos_path):
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
        query_videos_dir: where to look for the actual mp4 files of the video queries
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
            # Remove .npy if present to get actual wav filename
            base_audio_name = os.path.splitext(audio_file)[0]
            audio_source = os.path.join(audios_path, f"{base_audio_name}.wav")
            ranked_filename = f"{base_audio_name}_rank{index}.wav"
            dest_path = os.path.join(video_folder, ranked_filename)

            if os.path.exists(audio_source):
                shutil.copy(audio_source, dest_path)
            else:
                print(f"⚠️ Audio file {base_audio_name}.wav not found for {video_name}, skipping.")

    print(f"☑️ Inference examples created in {inference_dir}")



def evaluate_inference(infer_dict, k_values=[1, 5, 10]):
    total_videos = len(infer_dict)
    ranks = []

    for vid, audios in infer_dict.items():
        audio_ids = list(audios.keys())
        try:
            rank = audio_ids.index(vid) + 1
            ranks.append(rank)
        except ValueError:
            ranks.append(float('inf'))

    results = {"Total Videos": float(total_videos)}

    # Calculate Recall for whatever K values you want
    for k in k_values:
        # If the user only retrieved 3 audios, R@5 will just be the same as R@3
        recall = sum(1 for r in ranks if r <= k) / total_videos
        results[f"Recall@{k}"] = round(recall, 4)

    # MRR (Mean Reciprocal Rank)
    mrr = sum(1/r if r != float('inf') else 0 for r in ranks) / total_videos
    results["MRR"] = round(mrr, 4)

    return results



if __name__ == "__main__":
    vids = ["--XInAaMS6k", "-0gYWIOfqdM", "-3M-k4nIYIM", "-4ItJ9yTz_c", "-4o0jRbgHr4", "-4rdRn-FRXo", "-6lkiUAf_cQ", "-6VFTlZsft4"]
   
    # for vid in vids:
    #     print(",\n")
    #     pprint.pp(infer_similar_audio(query= vid, top_k=5, single_mode=True, random_sample_count=3, temp= 0.01), sort_dicts=False)
    #     print(",\n")

    # random video
    # extract_frames_from_video("/home/hossein/Desktop/B.sc.-project/inferred_examples/-0gYWIOfqdM/-0gYWIOfqdM.mp4","/home/hossein/Desktop/randomclip", 5)
    # extract_video_embedding("/home/hossein/Desktop/randomclip", "/home/hossein/Desktop/rando.npy")
    # pprint.pp(infer_similar_audio("/home/hossein/Desktop/rando.npy", 5, True,1, 0.01), sort_dicts=False)
    # pprint.pp(infer_similar_audio(query= "-0gYWIOfqdM", top_k=5, single_mode=True, random_sample_count=3, temp= 0.01), sort_dicts=False)
   
    # videos_training = vggsound_training_videos_generator("vggsound.csv", 3360, 1000, "train")
    # vggsound_batch_down_trim(videos_training, full_videos_path, trimmed_videos_path, download_and_trim_log_file, clip_length=10)
    # check_integrity()
    ALL_EMBS_FILES_TUPLE = load_all_normed_embeddings(audio_embeddings_path)
    # pprint.pp(infer_similar_audio(query= "-0gYWIOfqdM", top_k=5, single_mode=True, random_sample_count=3, temp= 0.01), sort_dicts=False)
    # pprint.pp(infer_similar_audio_fast(query= "-0gYWIOfqdM",all_emb_files_tuple= ALL_EMBS_FILES_TUPLE, top_k=5, single_mode=True, random_sample_count=3, temp= 0.01), sort_dicts=False)
    # example = infer_similar_audio_ultra_fast(query_lst= ["--XInAaMS6k", "-0gYWIOfqdM", "-3M-k4nIYIM"],all_emb_files_tuple=ALL_EMBS_FILES_TUPLE, top_k=5, random_sample=False, random_sample_count=3, temp= 0.01)
    # pprint.pp(example, sort_dicts=False)
    # create_inference_example(example, inference_dir=r"C:\Users\hosse\Desktop\rightNowjustTesting")

    # print(cosine_similarity(np.load(os.path.join(video_embeddings_path, "-3M-k4nIYIM.npy")).squeeze(), 
    #                         np.load(os.path.join(audio_embeddings_path, "-3M-k4nIYIM.npy")).squeeze()))
    # print(cosine_similarity(np.load(os.path.join(video_embeddings_path, "-3M-k4nIYIM.npy")).squeeze(), 
    #                         np.load(os.path.join(audio_embeddings_path, "-9whJW7BUSU.npy")).squeeze()))   

    # with keep.running():    
    #     vggsound_extract_audio_from_videos_batch(trimmed_videos_dir= trimmed_videos_path, audios_dir= audios_path, log_file=extract_audio_log_file)
    #     vggsound_extract_audio_embeddings_batch(audios_dir=audios_path, embeddings_dir= audio_embeddings_path, log_file=embeddings_audio_log_file)
    #     vggsound_batch_extract_frames_from_vids(trimmed_videos_dir= trimmed_videos_path, frames_dir= video_frames_path)
    #     vggsound_extract_video_embeddings_batch(frames_dir= video_frames_path, video_embeddings_dir= video_embeddings_path, log_file= video_embeddings_log_file)    
    
    # print("======== Inference and Evaluation ========")

    # sample_count = 3000
    # if not all_video_files:
    #     raise FileNotFoundError("No video embeddings found.")
    # count = min(sample_count, len(all_video_files))
    # final_keys = all_video_files[:count]
    # result = infer_similar_audio_ultra_fast(query_lst=final_keys, 
    #                                         all_emb_files_tuple=ALL_EMBS_FILES_TUPLE, 
    #                                         top_k=sample_count, 
    #                                         random_sample=False, 
    #                                         random_sample_count=0, 
    #                                         temp= 0.01)
    # metrics = evaluate_inference(result, k_values=[1, 5, 10, 50])
    # print("Evaluation Metrics:")
    # pprint.pp(metrics, sort_dicts=False)
    all_video_files = [
    f for f in os.listdir(video_embeddings_path)
    if f.endswith(".npy")
]

    all_video_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(video_embeddings_path, f)),
        reverse=True
    )

    all_video_files = [os.path.splitext(f)[0] for f in all_video_files][:10]

    
    # pprint.pp(result_old, sort_dicts=False)
    # pprint.pp(result_ult, sort_dicts=False)
    # pprint.pp(evaluate_inference(result, k_values=[1, 5, 10]), sort_dicts=False)
    # create_inference_example(result)