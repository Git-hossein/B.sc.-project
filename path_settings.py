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
        else:
            self.base_path = self._linux_base_path
            self.ffmpeg_path = "ffmpeg"

        # build paths initially
        self._build_paths()

    # -------------------------
    # internal helper
    # -------------------------
    def _build_paths(self):
        self.full_videos_path = os.path.join(self.base_path, "full_videos")
        self.trimmed_videos_path = os.path.join(self.base_path, "trimmed_videos")
        self.video_frames_path = os.path.join(self.base_path, "frames")
        self.audios_path = os.path.join(self.base_path, "audios")
        self.audio_embeddings_path = os.path.join(self.base_path, "audio_embeddings")
        self.video_embeddings_path = os.path.join(self.base_path, "video_embeddings")
        self.inferred_example_path = "./inferred_examples"
        self.TCML_server_input = os.path.join(self._TCML_server_base_path, "input")
        self.TCML_server_output_generated = os.path.join(self._TCML_server_base_path, "output/generated_audios")

        # Logs (same for both systems)
        self.download_and_trim_log_file = "./Logs_download_trim.csv"
        self.extract_audio_log_file = "./Logs_extract_audio.csv"
        self.embeddings_audio_log_file = "./Logs_audio_embeddings.csv"
        self.video_embeddings_log_file = "./Logs_video_embeddings.csv"
        self.vggsound_path = "./vggsound.csv"



paths_config = _PathSettings()