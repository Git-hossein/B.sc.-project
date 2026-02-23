import os

class PathSettings:
    def __init__(self):
        # Define both base paths
        self._windows_base_path = r"D:\Bsc.Thesis_Datasets\vggsound"
        self._linux_base_path = "/mnt/d/Bsc.Thesis_Datasets/vggsound"   # <- placeholder, you fill later

        # Default system (you can change this)
        self.base_path = self._windows_base_path

        # build paths initially
        self.set_to_windows_paths()

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

        # Logs (same for both systems)
        self.download_and_trim_log_file = "./Logs_download_trim.csv"
        self.extract_audio_log_file = "./Logs_extract_audio.csv"
        self.embeddings_audio_log_file = "./Logs_audio_embeddings.csv"
        self.video_embeddings_log_file = "./Logs_video_embeddings.csv"

    # -------------------------
    # public switch methods
    # -------------------------
    def set_to_windows_paths(self):
        self.base_path = self._windows_base_path
        self._build_paths()

    def set_to_linux_paths(self):
        self.base_path = self._linux_base_path
        self._build_paths()