import os
import platform
from typing import Literal
from dataclasses import dataclass

@dataclass
class _DatasetPaths:
    full_videos: str
    trimmed_videos: str
    frames: str
    audios: str
    cav_mae_audio: str
    cav_mae_video_clip: str
    cav_mae_video_frame: str
    cav_mae_sync_video: str
    cav_mae_sync_audio: str
    audio_embeddings: str
    video_embeddings: str
    video_embeddings_golden_frame: str
    video_embeddings_attention: str
    video_emb_log: str
    audio_emb_log: str
    audio_log: str
    down_trim_log: str



class _PathSettings:
    def __init__(self):

        
        self._windows_base_path = r"D:\Bsc.Thesis_Datasets\vggsound"
        self._linux_base_path = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound"
        self._TCML_server_base_path = "./tcml_exchange"

        if platform.system() == "Windows":
            self.base_path = self._windows_base_path
            self.ffmpeg_path = (
                r"C:\Users\hosse\Downloads\ffmpeg-8.0-essentials_build"
                r"\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
            )
            self.experiments_base_path = r"D:\Bsc.Thesis_Datasets\experiments"

        else:
            self.base_path = self._linux_base_path
            self.ffmpeg_path = "ffmpeg"
            self.experiments_base_path = (
                "/media/hossein/H.s.wildwildwest/"
                "Bsc.Thesis_Datasets/experiments"
            )

        self._build_paths()

    # -------------------------
    # internal helper
    # -------------------------
    def _build_paths(self):

        # -------------------------
        # General paths
        # -------------------------

        self.inferred_example_path = "./inferred_examples"

        self.TCML_server_input = os.path.join(
            self._TCML_server_base_path,"input")

        self.TCML_server_output_generated = os.path.join(
            self._TCML_server_base_path, "output/generated_audios")

        self.experiments_seen_path = os.path.join(
            self.experiments_base_path, "seen")

        self.experiments_unseen_path = os.path.join(
            self.experiments_base_path, "unseen")

        
        # -------------------------
        # Dataset paths
        # -------------------------

        self.training = self._build_dataset_paths("training")
        self.validation = self._build_dataset_paths("validation")
        self.test = self._build_dataset_paths("test")
        self.vggsound_path = "./vggsound/vggsound.csv"


    def _build_dataset_paths(self, split: Literal['training', 'test', 'validation']) -> _DatasetPaths:
        """
        Build all paths belonging to one dataset split.

        split:
            "training", "validation", or "test"
        """

        video_dir = "video"
        audio_dir = "audio"
        dataset_dir = "vggsound"
        split_path = os.path.join(self.base_path, split)

        return _DatasetPaths(
            full_videos=os.path.join(
                split_path,
                "full_videos",
            ),

            trimmed_videos=os.path.join(
                split_path,
                "trimmed_videos",
            ),

            frames=os.path.join(
                split_path,
                "frames",
            ),

            audios=os.path.join(
                split_path,
                "audios",
            ),

            cav_mae_audio=os.path.join(
                split_path,
                "cav_mae/audio_embeddings",
            ),

            cav_mae_video_clip=os.path.join(
                            split_path,
                            "cav_mae/video_embeddings/10FrameClipLevel",
                        ),

            cav_mae_video_frame=os.path.join(
                                        split_path,
                                        "cav_mae/video_embeddings/10FrameFrameLevel",
                                    ),

            cav_mae_sync_video=os.path.join(
                split_path,
                "cav_mae_sync/video_frame_embeddings",
            ),

            cav_mae_sync_audio=os.path.join(
                split_path,
                "cav_mae_sync/audio_frame_embeddings",
            ),

            audio_embeddings=os.path.join(
                split_path,
                "audio_embeddings",
            ),

            video_embeddings=os.path.join(
                split_path,
                "video_embeddings",
            ),

             video_embeddings_golden_frame= os.path.join(
                split_path,
                "video_embeddings_golden_8frames",
            ),

            video_embeddings_attention=
            os.path.join(
                split_path,
                "video_embeddings_attention",
            ),

            video_emb_log = os.path.join(
                video_dir,
                f"{split}_vid_emb_log"
            ),

            audio_emb_log = os.path.join(
                audio_dir,
                f"{split}_audio_emb_log"
            ),

            audio_log = os.path.join(
                audio_dir,
                f"{split}_audio_log"
            ),

            down_trim_log = os.path.join(
                dataset_dir,
                f"{split}_down_trim_log"
            ),
        )

paths_config = _PathSettings()
