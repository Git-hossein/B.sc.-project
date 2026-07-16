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
from generation_server import run_parallel_tcml_pipeline_continuation, evaluate_generation
from generation_server2 import run_parallel_tcml_pipeline_generation

class ExperimentTypes(Enum):
    generation = 0
    inference = 1
    dual = 2

class DatasetType(Enum):
    seen = 0
    unseen = 1

class Experiment:
    # Class-level mappings to keep the code dry and easy to update
    CONFIG_FILENAME = "config.json"
    EXPERIMENTS_BASE_DIR = paths_config.experiments_base_path
    RESULTS_FILENAME = "experiment_result.json"
    GENERATED_AUDIO_DIRNAME = "generated_audios"
    GENERATED_EMB_DIRNAME = "generated_audio_embeddings"
    LOGS_FILENAME = "logs_audio_embeddings.csv"
    ID_FILE = os.path.join(EXPERIMENTS_BASE_DIR, "ids.json") #TODO: make sure i have a file where i save the ids to make sure there are no duplicates
    _id_set = set()
    ALL_EMBS_FILES_TUPLE = load_all_normed_embeddings(paths_config.audio_embeddings_path)

    EXP_MAP = {
        "inference only": ExperimentTypes.inference,
        "generation only": ExperimentTypes.generation,
        "generation based on inference": ExperimentTypes.dual
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
        config_path = os.path.join(experiment_dir, cls.CONFIG_FILENAME)

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
            obj._with_text_descr = gp["conditioned on text prompts"] == "Yes"

        # Load results if they exist
        results_path = os.path.join(experiment_dir, cls.RESULTS_FILENAME)
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                res = json.load(f)
            obj._inference_dict = res.get("inference results")
            obj._generated_audio_folder = res.get("generation folder")
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
        self._k, self._t = -1, -1
        self._prompt_duration, self._top_mix, self._cfg_coef = -1, -1, -1
        self._with_text_descr, self._weight_by = None, None
        
        self._inference_params_set = False
        self._generation_params_set = False
        self._ran_inference = False
        self._ran_generation = False
        self._inference_dict = None
        self._generated_audio_folder = None
        self._inference_eval_results = None
        self._generation_eval_results = None
        
        self._build_paths()

    def _build_paths(self):
        base = paths_config.experiments_seen_path if self._dataset_type == DatasetType.seen else paths_config.experiments_unseen_path
        self._experiment_folder = os.path.join(base, f"experiment_{self._experiment_id}")
        os.makedirs(self._experiment_folder, exist_ok=True)
        if self._experiment_type in (ExperimentTypes.generation, ExperimentTypes.dual):

            self._generated_audio_folder = os.path.join(self._experiment_folder, self.GENERATED_AUDIO_DIRNAME)
            self._generated_audio_embeddings_dir = os.path.join(self._experiment_folder, self.GENERATED_EMB_DIRNAME)
            self._logs_audio_embeddings = os.path.join(self._experiment_folder, self.LOGS_FILENAME)
            
            os.makedirs(self._generated_audio_folder, exist_ok=True)
            os.makedirs(self._generated_audio_embeddings_dir, exist_ok=True)
            
            if not os.path.exists(self._logs_audio_embeddings):
                with open(self._logs_audio_embeddings, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(["video_id", "audio_embedding_status"])

    def save_config(self, special_notes=None, overwrite=False):
        config_path = os.path.join(self._experiment_folder, self.CONFIG_FILENAME)
        
        if overwrite or not os.path.exists(config_path): 

            is_inf = self._experiment_type in (ExperimentTypes.inference, ExperimentTypes.dual)
            is_gen = self._experiment_type in (ExperimentTypes.generation, ExperimentTypes.dual)

            type_str = "inference only"
            if self._experiment_type == ExperimentTypes.dual: type_str = "generation based on inference"
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
                    "conditioned on text prompts": "Yes" if self._with_text_descr else "No",
                    "weight used for mixing": self._weight_by
                } if is_gen else "None",
                "video_list": self._videos,
                "notes": special_notes or "None"
            }
            
            with open(config_path, "w") as f:
                json.dump(config_file, f, indent=4)

    def save_experiment(self, override=False):
        experiment_results_path = os.path.join(self._experiment_folder, self.RESULTS_FILENAME)
        
        if override or not os.path.exists(experiment_results_path): 
            type_str = "inference only"
            if self._experiment_type == ExperimentTypes.dual: type_str = "generation based on inference"
            elif self._experiment_type == ExperimentTypes.generation: type_str = "generation only"
            results = {
                "experiment ID": self._experiment_id,
                "dataset type": "training" if self._dataset_type == DatasetType.seen else "test",
                "experiment type": type_str,
                "generation folder": self._generated_audio_folder,
                "inference evaluation": self._inference_eval_results,
                "generation evaluation": self._generation_eval_results
            }
            with open(os.path.join(self._experiment_folder, self.RESULTS_FILENAME), "w") as f:
                json.dump(results, f, indent=4)

    def set_generation_config(self, cfg_coef, prompt_duration, top_mix, with_text_descr, weight_by: Literal["softmax_score", "cosine_sim"]):
        if self._experiment_type not in (ExperimentTypes.generation, ExperimentTypes.dual): 
            raise ValueError(f"Experiment type {self._experiment_type} does not support generation.")
        self._cfg_coef, self._prompt_duration = cfg_coef, prompt_duration
        self._top_mix, self._with_text_descr = top_mix, with_text_descr
        self._weight_by = weight_by
        self._generation_params_set = True

    def set_inference_config(self, k, t):
        if self._experiment_type not in (ExperimentTypes.inference, ExperimentTypes.dual): 
            raise ValueError(f"Experiment type {self._experiment_type} does not support inference.")
        self._k, self._t = k, t
        self._inference_params_set = True


    def run_evaluations(self, overwrite= False):

        if self._experiment_type in (ExperimentTypes.inference,):
            self._inference_eval_results = evaluate_inference(self._inference_dict)
            self.save_config()
            self.save_experiment()

        if self._experiment_type in (ExperimentTypes.dual, ExperimentTypes.generation):
            # Gather files
            og_files, gen_files, mix_files = [], [], []
            for file in os.listdir(self._generated_audio_folder):
                if file.endswith("_RAW_MIX.wav"): 
                    mix_files.append(file)
                elif file.endswith("_GEN.wav"): 
                    gen_files.append(file)
                    og_files.append(file.removesuffix("_GEN.wav"))

            assert len(og_files) == len(gen_files), f"the og files have {len(og_files)} != {len(gen_files)}"
            if len(mix_files) != 0: assert len(og_files) == len(mix_files)
            assert set(og_files) == set([Experiment._strip_ext(f) for f in self._videos])
            # Metrics alignment
            res_path = os.path.join(self._experiment_folder, self.RESULTS_FILENAME)
            if overwrite or not os.path.exists(res_path):
                print("📊 Computing evaluation metrics...")
                vggsound_extract_audio_embeddings_batch(self._generated_audio_folder, self._generated_audio_embeddings_dir, None, self._logs_audio_embeddings)

                og_embs_dir = paths_config.audio_embeddings_path if self._dataset_type == DatasetType.seen else paths_config.test_audio_embeddings_path
                og_embs = load_all_normed_embeddings(og_embs_dir, self._videos)
                gen_embs = load_all_normed_embeddings(self._generated_audio_embeddings_dir, gen_files)
                if len(mix_files) !=0:                 
                    mix_embs = load_all_normed_embeddings(self._generated_audio_embeddings_dir, mix_files)

                if self._dataset_type == DatasetType.seen: 
                    self._inference_eval_results = evaluate_inference(self._inference_dict)
                else:
                    self._inference_eval_results = None
                self._generation_eval_results = { 
                    "audiogen eval": evaluate_generation(gen_embs, og_embs)
                }
                if len(mix_files) != 0: self._generation_eval_results['naive mix eval'] = evaluate_generation(mix_embs, og_embs)
                
                self.save_config()
                self.save_experiment()
                

    def load_inference_manual(self, inference_dict):
        if self._experiment_type not in (ExperimentTypes.dual, ExperimentTypes.inference):
            raise Exception(f"cannot load inference results for experiment type: {self._experiment_type}")
        print("you are manually loading the inference results. make sure to set the inference params if needed")
        self._ran_inference = True
        self._inference_dict = inference_dict

    def run_inference(self, overwrite = False):
        if not self._ran_inference or overwrite:
            if self._experiment_type in (ExperimentTypes.dual, ExperimentTypes.inference):     
                query_dir = paths_config.test_video_embeddings_path if self._dataset_type== DatasetType.unseen else paths_config.video_embeddings_path
                print(f"... running inference for experiment{self._experiment_id}...")
                self._inference_dict = infer_similar_audio_ultra_fast_ultimate(
                    query_lst=self._videos, 
                    query_dir= query_dir,
                    all_emb_dict=self.ALL_EMBS_FILES_TUPLE, 
                    top_k=self._k,
                    random_sample = False,
                    temp=self._t
                )
                assert len(self._inference_dict) == len(self._videos), f" {len(self._inference_dict)} != {len(self._videos)}"
                self._ran_inference = True
                                
    def load_generation_manual(self, generation_folder):
        if self._experiment_type not in (ExperimentTypes.dual, ExperimentTypes.generation):
            raise Exception(f"cannot load generation results for experiment type: {self._experiment_type}")
        print("you are manually loading the generation results. make sure to set the generation params if needed")

        if not os.path.exists(generation_folder):
            raise FileNotFoundError(f"Source folder does not exist: {generation_folder}")
        
        src_abs = os.path.abspath(generation_folder)
        dst_abs = os.path.abspath(self._generated_audio_folder)
        if src_abs == dst_abs:
            print("Source and destination are the same. Skipping copy.")
        elif len(os.listdir(dst_abs)) != 0:
            print(f"the generation folder {dst_abs} seems to already contain some files. Skipping copy.")  
        else:
            print(f"Copying contents: {generation_folder} -> {self._generated_audio_folder}")
            # dirs_exist_ok is critical here because _build_paths already created the folder
            shutil.copytree(generation_folder, self._generated_audio_folder, dirs_exist_ok=True)

            
        self._ran_generation = True

    def run_generation(self, num_GPU, auto_download, overwrite = True):
        if not self._ran_generation or overwrite:

            if self._experiment_type == ExperimentTypes.dual:
                options = {
                    "cfg_coef": self._cfg_coef,
                    "prompt_duration": self._prompt_duration,
                    "with_text_descr": self._with_text_descr,
                    "num_audio_mix": self._top_mix,
                    "weight_by": "softmax_score"
                }
                print(f"... running generation for experiment{self._experiment_id}...")
                run_parallel_tcml_pipeline_continuation(self._inference_dict, self._generated_audio_folder, 
                                                        timeout_minutes=12*60, 
                                                        num_GPU= num_GPU, 
                                                        options=options, 
                                                        auto_download= auto_download)
                self._ran_generation = True
            
            if self._experiment_type == ExperimentTypes.generation:
                options = {"cfg_coef": self._cfg_coef}
                print(f"... running generation for experiment{self._experiment_id}...")
                run_parallel_tcml_pipeline_generation(video_list= self._videos, 
                                                      local_down_dst= self._generated_audio_folder, 
                                                      timeout_minutes=12*60, 
                                                      num_GPU= num_GPU, 
                                                      options=options, 
                                                      auto_download=auto_download)
                self._ran_generation = True

if __name__ == "__main__":


#     my_experiment = ExperimentSeen(1,all_video_files,  ExperimentTypes.dual, DatasetType.unseen)
#     my_experiment.set_generation_config(3.0, 2, 5, True)
#     my_experiment.set_inference_config(5, 0.01)
    test_vids = [vid for vid in os.listdir(paths_config.test_video_embeddings_path)]
    train_vids = [vid for vid in os.listdir(paths_config.video_embeddings_path)]
    # my_dict_test = infer_similar_audio_ultra_fast_ultimate(test_vids, paths_config.test_video_embeddings_path, Experiment.ALL_EMBS_FILES_TUPLE, 2, False, 1, 0.01)
    # my_dict = infer_similar_audio_ultra_fast_ultimate(train_vids, paths_config.video_embeddings_path, Experiment.ALL_EMBS_FILES_TUPLE, len(train_vids), False, 1, 0.01)
    # my_experiment0 = Experiment(0, train_vids, ExperimentTypes.inference, DatasetType.seen)
    # my_experiment0.set_inference_config(k = len(train_vids), t=0.01)
    # my_experiment0.run_inference()
    # my_experiment0.run_evaluations()


    # ----
    # my_experiment1 = Experiment(1, test_vids, ExperimentTypes.dual, DatasetType.unseen)
    # my_experiment1.set_inference_config(k = 5, t=0.01)
    # my_experiment1.set_generation_config(3.0, 2, 5, True, "softmax_score")
    # my_experiment1.run_inference()
    # my_experiment1.load_generation_manual("./generated_examples")
    # my_experiment1.run_evaluations()


    # ----
    # my_experiment2 = Experiment(2, test_vids, ExperimentTypes.dual, DatasetType.unseen)
    # my_experiment2.set_inference_config(k = 5, t=0.01)
    # my_experiment2.set_generation_config(3.0, 8, 5, True, "softmax_score")
    # my_experiment2.run_inference()
    # my_experiment2.load_generation_manual("./generated_examples_2")
    # my_experiment2.run_evaluations()


    # ----
    # my_experiment3 = Experiment(3, test_vids, ExperimentTypes.generation, DatasetType.unseen)
    # my_experiment3.set_generation_config(3.0, -1, -1, True, None)
    # my_experiment3.load_generation_manual("./generated_examples_3")
    # my_experiment3.run_evaluations(overwrite=True)
    

    # ----

    my_experiment4 = Experiment(4, ["-0BIyqJj9ZU", "-0jeONf82dE", "-0p7hKXZ1ww", "-0pJqpNjft4"], ExperimentTypes.generation, DatasetType.unseen)
    my_experiment4.set_generation_config(3.0, -1, -1, True, None)
    my_experiment4.load_generation_manual("./yolo")
    my_experiment4.run_evaluations(overwrite=True)
    