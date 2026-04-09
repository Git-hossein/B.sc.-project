from typing import Literal, Optional, List, Union, Dict
from enum import Enum
import pprint
import os
import numpy as np
import csv
import json
import time
import warnings
import shutil
from path_settings import paths_config
from inference import infer_similar_audio_ultra_fast_ultimate, load_all_normed_embeddings, evaluate_inference
from audio_processing import vggsound_extract_audio_embeddings_batch
from generation_server import run_tcml_audio_pipeline, evaluate_generation

class ExperimentTypes(Enum):
    generation = 0
    inference = 1
    dual = 2

class DatasetType(Enum):
    seen = 0
    unseen = 1

class ExperimentSeen:
    # Class-level mappings to keep the code dry and easy to update
    EXP_MAP = {
        "inference only": ExperimentTypes.inference,
        "generation only": ExperimentTypes.generation,
        "inference and generation": ExperimentTypes.dual
    }
    DS_MAP = {"training": DatasetType.seen, "test": DatasetType.unseen}

    @staticmethod
    def _strip_ext(f: str) -> str:
        return os.path.splitext(f)[0]

    @classmethod
    def load_experiment(cls, experiment_id: str, type: Optional[DatasetType] = None):
        """Reconstructs the experiment object from existing disk files."""
        candidate_dirs = []
        folder_name = f"experiment_{experiment_id}"

        if type:
            base = paths_config.experiments_seen_path if type == DatasetType.seen else paths_config.experiments_unseen_path
            candidate_dirs.append(os.path.join(base, folder_name))
        else:
            candidate_dirs.append(os.path.join(paths_config.experiments_seen_path, folder_name))
            candidate_dirs.append(os.path.join(paths_config.experiments_unseen_path, folder_name))

        existing_dirs = [d for d in candidate_dirs if os.path.exists(d)]

        if not existing_dirs:
            print(f"❌ No experiment '{experiment_id}' found.")
            return None
        if len(existing_dirs) > 1:
            print(f"⚠️ Multiple experiments for '{experiment_id}' found. Specify a path.")
            return None

        experiment_dir = existing_dirs[0]
        config_path = os.path.join(experiment_dir, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config missing for {experiment_id}")

        with open(config_path, 'r') as f:
            c = json.load(f)

        # Initialize instance
        obj = cls(
            experiment_id=c["experiment ID"],
            videos=c["video_list"],
            experiment_type=cls.EXP_MAP[c["experiment type"]],
            dataset_type=cls.DS_MAP[c["dataset type"]]
        )

        # Restore parameters
        if c["inference_params"] != "None":
            obj._k = int(c["inference_params"]["K (retrieved audio)"])
            obj._t = c["inference_params"]["T (softmax_temp)"]

        if c["generation_params"] != "None":
            gp = c["generation_params"]
            obj._cfg_coef = gp["cfg_coef"]
            obj._prompt_duration = gp["audio prompt_duration"]
            obj._top_mix = gp["num of audio samples used"]
            obj._with_descr = gp["conditioned on text prompts"] == "Yes"

        # Load results if they exist
        results_path = os.path.join(experiment_dir, "experiment result.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                res = json.load(f)
            obj._inference_dict = res.get("inference results")
            obj._generated_audio = res.get("generation folder")
            obj._inference_eval_results = res.get("inference evaluation")
            obj._generation_eval_results = res.get("generation evaluation")

        return obj

    def __init__(self, experiment_id: str, videos: Union[List[str], str], 
                 experiment_type: ExperimentTypes, dataset_type: DatasetType):
        
        self._experiment_id = experiment_id
        # Ensure videos is always a list
        self._videos = videos if isinstance(videos, list) else [videos]
        self._experiment_type = experiment_type
        self._dataset_type = dataset_type
 
        # Defaults
        self._k, self._t = 5, 0.01
        self._prompt_duration, self._top_mix, self._cfg_coef = 2, 5, 3.0
        self._with_descr = True
        
        self._inference_dict = None
        self._generated_audio = None
        self._inference_eval_results = None
        self._generation_eval_results = None
        
        self._build_paths()

    def _build_paths(self):
        base = paths_config.experiments_seen_path if self._dataset_type == DatasetType.seen else paths_config.experiments_unseen_path
        self._experiment_folder = os.path.join(base, f"experiment_{self._experiment_id}")
        self._generated_audio_folder = os.path.join(self._experiment_folder, "generated_audios")
        self._generated_audio_embeddings_dir = os.path.join(self._experiment_folder, "generated_audio_embeddings")
        self._logs_audio_embeddings = os.path.join(self._experiment_folder, "Logs_audio_embeddings.csv")
        
        os.makedirs(self._generated_audio_folder, exist_ok=True)
        os.makedirs(self._generated_audio_embeddings_dir, exist_ok=True)
        
        if not os.path.exists(self._logs_audio_embeddings):
            with open(self._logs_audio_embeddings, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["video_id", "audio_embedding_status"])

    def save_config(self, special_notes=None, override=False):
        if not override: return

        is_inf = self._experiment_type in (ExperimentTypes.inference, ExperimentTypes.dual)
        is_gen = self._experiment_type in (ExperimentTypes.generation, ExperimentTypes.dual)

        type_str = "inference only"
        if self._experiment_type == ExperimentTypes.dual: type_str = "inference and generation"
        elif self._experiment_type == ExperimentTypes.generation: type_str = "generation only"

        config_file = {
            "experiment ID": self._experiment_id,
            "time": time.ctime(),
            "number of videos": len(self._videos),
            "dataset type": "training" if self._dataset_type == DatasetType.seen else "test",
            "experiment type": type_str,
            "inference_params": {
                "K (retrieved audio)": self._k,
                "T (softmax_temp)": self._t
            } if is_inf else "None",
            "generation_params": {
                "cfg_coef": self._cfg_coef,
                "audio prompt_duration": self._prompt_duration,
                "num of audio samples used": self._top_mix,
                "conditioned on text prompts": "Yes" if self._with_descr else "No"
            } if is_gen else "None",
            "video_list": self._videos,
            "notes": special_notes or "None"
        }
        
        with open(os.path.join(self._experiment_folder, "config.json"), "w") as f:
            json.dump(config_file, f, indent=4)

    def save_experiment(self, override=False):
        if not override: return

        results = {
            "experiment ID": self._experiment_id,
            "inference results": self._inference_dict,
            "generation folder": self._generated_audio,
            "inference evaluation": self._inference_eval_results,
            "generation evaluation": self._generation_eval_results
        }
        with open(os.path.join(self._experiment_folder, "experiment result.json"), "w") as f:
            json.dump(results, f, indent=4)

    def set_generation_config(self, cfg_coef, prompt_duration, top_mix, with_descr):
        if self._experiment_type not in (ExperimentTypes.generation, ExperimentTypes.dual): 
            raise ValueError(f"Experiment type {self._experiment_type} does not support generation.")
        self._cfg_coef, self._prompt_duration = cfg_coef, prompt_duration
        self._top_mix, self._with_descr = top_mix, with_descr

    def set_inference_config(self, k, t):
        # FIX: Check for inference or dual, not generation!
        if self._experiment_type not in (ExperimentTypes.inference, ExperimentTypes.dual): 
            raise ValueError(f"Experiment type {self._experiment_type} does not support inference.")
        self._k, self._t = k, t

    def run_experiments(self, override=False):
        if override:
            self._generated_audio, self._inference_dict = None, None
            shutil.rmtree(self._generated_audio_folder, ignore_errors=True)
            shutil.rmtree(self._generated_audio_embeddings_dir, ignore_errors=True)
            if os.path.exists(self._logs_audio_embeddings): os.remove(self._logs_audio_embeddings)
            self._build_paths()

        self._run_inference()
        self._run_generation()

        # Gather files
        og_files, gen_files, mix_files = [], [], []
        for file in os.listdir(self._generated_audio_folder):
            if file.endswith("_RAW_MIX.wav"): 
                mix_files.append(file)
            elif file.endswith("_GEN.wav"): 
                gen_files.append(file)
                og_files.append(file.removesuffix("_GEN.wav"))

        assert len(og_files) == len(gen_files) == len(mix_files)
        assert set(og_files) == set([ExperimentSeen._strip_ext(f) for f in self._videos])
        # Metrics alignment
        res_path = os.path.join(self._experiment_folder, "experiment result.json")
        if override or not os.path.exists(res_path):
            print("📊 Computing evaluation metrics...")
            vggsound_extract_audio_embeddings_batch(self._generated_audio_folder, self._generated_audio_embeddings_dir, None, self._logs_audio_embeddings)

            og_embs = load_all_normed_embeddings(paths_config.audio_embeddings_path, self._videos)
            mix_embs = load_all_normed_embeddings(self._generated_audio_embeddings_dir, mix_files)
            gen_embs = load_all_normed_embeddings(self._generated_audio_embeddings_dir, gen_files)

            self._inference_eval_results = evaluate_inference(self._inference_dict)
            self._generation_eval_results = {
                "naive mix eval": evaluate_generation(mix_embs, og_embs), 
                "audiogen eval": evaluate_generation(gen_embs, og_embs)
            }
            
            self.save_config(override=True)
            self.save_experiment(override=True)

        pprint.pp(self._inference_eval_results, sort_dicts=False)
        pprint.pp(self._generation_eval_results, sort_dicts=False)

    def _run_inference(self):
        if self._inference_dict is None:
            db = load_all_normed_embeddings(paths_config.audio_embeddings_path)
            self._inference_dict = infer_similar_audio_ultra_fast_ultimate(
                query_lst=self._videos, 
                all_emb_dict=db, 
                top_k=self._k,
                random_sample = False,
                temp=self._t
            )
                                
    def _run_generation(self):
        if self._generated_audio is None:
            self._generated_audio = self._generated_audio_folder
            run_tcml_audio_pipeline(self._inference_dict, self._generated_audio_folder)