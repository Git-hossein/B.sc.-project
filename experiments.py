from typing import Literal, Optional
from enum import Enum
import pprint
from path_settings import paths_config
from inference import infer_similar_audio_ultra_fast_ultimate, load_all_normed_embeddings, evaluate_inference
from audio_processing import vggsound_extract_audio_embeddings_batch
from generation_server import run_tcml_audio_pipeline, evaluate_generation
import os
import numpy as np
import csv
import json
import time
import warnings


class ExperimentTypes(Enum):
    generation = 0
    inference = 1
    dual = 2

class DatasetType(Enum):
    seen = 0
    unseen = 1

class ExperimentSeen:


    @classmethod
    def load_experiment(cls, experiment_id, path=None):
        candidate_dirs = []

        if path is not None:
            candidate_dirs.append(os.path.join(path, f"experiment_{experiment_id}"))
        else:
            candidate_dirs.append(os.path.join(paths_config.experiments_seen_path, f"experiment_{experiment_id}"))
            candidate_dirs.append(os.path.join(paths_config.experiments_unseen_path, f"experiment_{experiment_id}"))

        # Check which directories actually exist
        existing_dirs = [d for d in candidate_dirs if os.path.exists(d)]

        if not existing_dirs:
            if path is not None:
                print(f"No experiment with the id '{experiment_id}' found in the given path.")
            else:
                print(f"No experiment with the id '{experiment_id}' found in either seen or unseen paths.")
            return 

        if len(existing_dirs) > 1:
            print(f"Multiple experiments with the id '{experiment_id}' found. Please specify a path.")
            return 

        # Only one valid experiment found
        experiment_dir = existing_dirs[0]

        # Load the configuration file
        config_path = os.path.join(experiment_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"no config file found for experiment with id: {experiment_id}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        exp_type_map = {
            "inference only": ExperimentTypes.inference,
            "generation only": ExperimentTypes.generation,
            "inference and generation": ExperimentTypes.dual
        }
        ds_type_map = {"training": DatasetType.seen, "test": DatasetType.unseen}

        obj = cls(
            experiment_id = config["experiment ID"],
            videos = config["video_list"],  # We now save the full list in config
            experiment_type = exp_type_map[config["experiment type"]],
            dataset_type = ds_type_map[config["dataset type"]],
        )
    
    
        # Restore parameters
        if config["inference_params"] != "None":
            obj._k = int(config["inference_params"]["K (retrieved audio)"])
            obj._t = config["inference_params"]["T (softmax_temp)"]

        if config["generation_params"] != "None":
            obj._cfg_coef = config["generation_params"]["cfg_coef"]
            obj._prompt_duration = config["generation_params"]["audio prompt_duration"]
            obj._top_mix = config["generation_params"]["num of audio samples used"]
            obj._with_descr = config["generation_params"]["conditioned on text prompts"] == "Yes"


        # Load the experiment reuslts file
        exp_results_path = os.path.join(experiment_dir, "experiment result.json")
        if os.path.exists(exp_results_path):
            with open(exp_results_path, 'r') as f:
                exp_results = json.load(f)
            
            obj._inference_dict = exp_results["inference results"]
            obj._generated_audio = exp_results["generation folder"]
            obj._inference_eval_results = exp_results["inference evaluation"]
            obj._generation_eval_results = exp_results["generation evaluation"]




        return obj

        

    def __init__(self, experiment_id, videos:list[str]|str ,experiment_type: ExperimentTypes, dataset_type: DatasetType):

        self._experiment_id = experiment_id
        self._videos = videos
        self._experiment_type = experiment_type
        self._dataset_type = dataset_type
 
        self._k = 5
        self._t = 0.01
        self._prompt_duration = 2
        self._top_mix = 5
        self._cfg_coef = 3.0
        self._with_descr = True
        self._inference_dict = None
        self._generated_audio = None
        self._build_paths()


    def save_config(self, special_notes = None, override = False):
        eval_avaiable = True if self._experiment_type in (ExperimentTypes.inference, ExperimentTypes.dual) else False
        gen_avaiable = True if self._experiment_type in (ExperimentTypes.generation, ExperimentTypes.dual) else False

        data_type = "training" if self._dataset_type is DatasetType.seen else "test"

        experiment_type = (
            "inference and generation"
            if self._experiment_type is ExperimentTypes.dual
            else "inference only"
            if self._experiment_type is ExperimentTypes.inference
            else "generation only"
        )

        config_file = {"experiment ID": self._experiment_id,
                       "time": time.ctime(),
                       "number of videos": len(self._videos),
                       "sepcial notes about video dataset": special_notes if special_notes else "None",
                       "dataset type": data_type,
                       "experiment type": experiment_type,
                       "inference_params": "None" if not eval_avaiable else
                       {
                           "K (retrieved audio)": self._k,
                           "T (softmax_temp)": self._t
                       },
                       "generation_params": "None" if not gen_avaiable else
                       {
                           "cfg_coef": self._cfg_coef,
                           "audio prompt_duration": self._prompt_duration,
                           "num of audio samples used": self._top_mix,
                           "conditioned on text prompts": "Yes" if self._with_descr else "No"
                       },
                       "video_list": self._videos
                       }
        
        if override:
            with open(os.path.join(self._experiment_folder, "config.json"), mode="w") as f:
                json.dump(config_file,f, indent= 4)

    def save_experiment(self, override = False):

        config_file = {
            "experiment ID": self._experiment_id,
            "inference results": self._inference_dict,
            "generation folder": self._generated_audio,
            "inference evaluation": self._inference_eval_results,
            "generation evaluation": self._generation_eval_results
            }
        if override:
            with open(os.path.join(self._experiment_folder, "experiment result.json"), mode="w") as f:
                json.dump(config_file,f, indent= 4)


    def set_generation_config(self, cfg_coef, prompt_duration, top_mix, with_descr):
        if self._experiment_type not in (ExperimentTypes.generation, ExperimentTypes.dual): 
            raise Exception(f"cannot set params for generation when the experiment type is {self._experiment_type}")
        self._cfg_coef = cfg_coef
        self._prompt_duration = prompt_duration
        self._top_mix = top_mix 
        self._with_descr = with_descr

    def set_inference_config(self, k, t):
        if self._experiment_type not in (ExperimentTypes.generation, ExperimentTypes.dual): 
            raise Exception(f"cannot set params for inference when the experiment type is {self._experiment_type}")
        self._k = k
        self._t = t

    def _run_inference(self):
        if self._inference_dict is None:
            all_emb_dict = load_all_normed_embeddings(paths_config.audio_embeddings_path)
            self._inference_dict = infer_similar_audio_ultra_fast_ultimate(query_lst=self._videos, 
                                                                           all_emb_dict= all_emb_dict,
                                                                            top_k= self._k,
                                                                            random_sample=False,
                                                                            temp= self._t)
                                
    
    def _run_generation(self):
        if self._generated_audio is None:
            self._generated_audio = self._generated_audio_folder
            run_tcml_audio_pipeline(self._inference_dict, self._generated_audio_folder)
    


    def _build_paths(self):

        dst_dir = paths_config.experiments_seen_path if self._dataset_type == DatasetType.seen else paths_config.experiments_unseen_path
        self._experiment_folder = os.path.join(dst_dir, f"experiment_{self._experiment_id}")
        self._generated_audio_folder = os.path.join(self._experiment_folder, "generated_audios")
        self._logs_audio_embeddings = os.path.join(self._experiment_folder, "Logs_audio_embeddings.csv")
        self._json_inference_dict = os.path.join(self._experiment_folder, "inference_dict.json")
        self._json_inference_eval_result = os.path.join(self._experiment_folder, "inference_eval.json")
        self._json_generation_eval_result = os.path.join(self._experiment_folder, "generation_eval.json")


        os.makedirs(self._experiment_folder, exist_ok=True)
        os.makedirs(self._generated_audio_folder, exist_ok=True)
        os.makedirs(self._logs_audio_embeddings, exist_ok=True)

        write_header = not os.path.exists(self._logs_audio_embeddings) 
        if write_header:
            with open(self._logs_audio_embeddings, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["video_id", "audio_embedding_status"])


    def run_experiments(self, override= False):
        # evaluate inferece using the specified params
        if override: self._generated_audio, self._inference_dict = None, None

        self._run_inference()
        self._run_generation()

        og_files= []
        gen_files = []
        mix_files = []
        rv_file_extension = lambda f: os.path.splitext(f)[0]

        for file in os.listdir(self._generated_audio_folder):


            if file.endswith("_RAW_MIX.wav"):
                mix_files.append(file)
                clean_file = file.replace("_RAW_MIX", "")
                og_files.append(rv_file_extension(clean_file))

            elif file.endswith("_GEN.wav"):
                gen_files.append(file)
                clean_file = file.replace("_GEN", "")
                og_files.append(rv_file_extension(clean_file))

            else:
                warnings.warn(f"unknown file {file} found", UserWarning)

        assert set(og_files) == set([rv_file_extension(f) for f in self._videos])


        results_path = os.path.join(self._experiment_folder, "experiment result.json")

        if override or not os.path.exists(results_path):

            vggsound_extract_audio_embeddings_batch(self._generated_audio_folder, self._generated_audio_folder, None, self._logs_audio_embeddings)

            og_embs_dict = load_all_normed_embeddings(paths_config.audio_embeddings_path, self._videos)
            naive_mix_embs_dict = load_all_normed_embeddings(self._generated_audio_folder, mix_files)
            gen_embs_dict = load_all_normed_embeddings(self._generated_audio_folder, gen_files)

            self._inference_eval_results = evaluate_inference(self._inference_dict)
            self._generation_eval_results = {"naive mix eval": evaluate_generation(naive_mix_embs_dict, og_embs_dict), 
                                            "audiogen eval": evaluate_generation(gen_embs_dict, og_embs_dict)}
            
            self.save_config(override=True)
            self.save_experiment(override=True)
            
  

        print("="*20)
        pprint.pp(self._inference_eval_results, sort_dicts=False)
        print("="*20)
        pprint.pp(self._generation_eval_results, sort_dicts=False)


    def print_eval_results(self, eval: Literal["generation", "inference"]):
        pprint.pp(self._inference_eval_results if eval is "inference" else self._generation_eval_results, sort_dicts=False)
