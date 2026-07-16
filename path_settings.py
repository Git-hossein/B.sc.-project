import os
import platform

class _PathSettings:
    def __init__(self):

        self._windows_base_path = r"D:\Bsc.Thesis_Datasets\vggsound"
        self._linux_base_path = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound"
        self._TCML_server_base_path = "./tcml_exchange"

        if platform.system() == "Windows":
            self.base_path = self._windows_base_path
            self.ffmpeg_path = r"C:\Users\hosse\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
            self.experiments_base_path = r"D:\Bsc.Thesis_Datasets\experiments"
        else:
            self.base_path = self._linux_base_path
            self.ffmpeg_path = "ffmpeg"
            self.experiments_base_path = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/experiments"

        # build paths initially
        self._build_paths()

    # -------------------------
    # internal helper
    # -------------------------
    def _build_paths(self):

        self.inferred_example_path = "./inferred_examples"
        self.TCML_server_input = os.path.join(self._TCML_server_base_path, "input")
        self.TCML_server_output_generated = os.path.join(self._TCML_server_base_path, "output/generated_audios")
        
        self.experiments_seen_path = os.path.join(self.experiments_base_path ,"seen")
        self.experiments_unseen_path = os.path.join(self.experiments_base_path ,"unseen")

        ########
        self.training_full_videos_path = os.path.join(self.base_path, "training" ,"full_videos")
        self.training_trimmed_videos_path = os.path.join(self.base_path, "training", "trimmed_videos")
        self.training_video_frames_path = os.path.join(self.base_path, "training", "frames")
        self.training_audios_path = os.path.join(self.base_path, "training", "audios")
        self.training_audio_embeddings_path = os.path.join(self.base_path, "training", "audio_embeddings")
        self.training_video_embeddings_path = os.path.join(self.base_path, "training", "video_embeddings")
        ########
        self.validation_full_videos_path = os.path.join(self.base_path, "validation", "full_videos")
        self.validation_trimmed_videos_path = os.path.join(self.base_path, "validation", "trimmed_videos")
        self.validation_video_frames_path = os.path.join(self.base_path, "validation", "frames")
        self.validation_audios_path = os.path.join(self.base_path, "validation", "audios")
        self.validation_audio_embeddings_path = os.path.join(self.base_path,"validation", "audio_embeddings")
        self.validation_video_embeddings_path = os.path.join(self.base_path,"validation", "video_embeddings")

        ########
        self.test_full_videos_path = os.path.join(self.base_path, "test", "full_videos")
        self.test_trimmed_videos_path = os.path.join(self.base_path, "test", "trimmed_videos")
        self.test_video_frames_path = os.path.join(self.base_path, "test", "frames")
        self.test_audios_path = os.path.join(self.base_path, "test", "audios")
        self.test_audio_embeddings_path = os.path.join(self.base_path,"test", "audio_embeddings")
        self.test_video_embeddings_path = os.path.join(self.base_path,"test", "video_embeddings")


        # Logs (same for both systems)

        video_dir = "video"
        audio_dir = "audio"
        dataset_dir = "vggsound"

        self.download_and_trim_log_file = dataset_dir + "/Logs_download_trim.csv"
        self.validation_download_and_trim_log_file = dataset_dir + "/Validation_Logs_download_trim.csv"
        self.test_download_and_trim_log_file = dataset_dir + "/Test_Logs_download_trim.csv"
        self.extract_audio_log_file = audio_dir + "/Logs_extract_audio.csv"
        self.validation_extract_audio_log_file = audio_dir + "/Validation_Logs_extract_audio.csv"
        self.test_extract_audio_log_file = audio_dir + "/Test_Logs_extract_audio.csv"
        self.embeddings_audio_log_file = audio_dir + "/Logs_audio_embeddings.csv"
        self.validation_embeddings_audio_log_file = audio_dir + "/Validation_Logs_audio_embeddings.csv"
        self.test_embeddings_audio_log_file = audio_dir + "/Test_Logs_audio_embeddings.csv"
        self.video_embeddings_log_file = video_dir + "/Logs_video_embeddings.csv"
        self.validation_video_embeddings_log_file = video_dir + "/Validation_Logs_video_embeddings.csv"
        self.test_video_embeddings_log_file = video_dir + "/Test_Logs_video_embeddings.csv"
        self.vggsound_path = dataset_dir + "/vggsound.csv"



paths_config = _PathSettings()