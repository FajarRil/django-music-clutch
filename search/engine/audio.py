import logging
import numpy as np
import os
import librosa
from scipy.signal import spectrogram
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AudioSimilarityEngine:
    """
    Handles audio-based similarity detection and fingerprinting
    """

    @staticmethod
    def load_local_audio(file_path):
        """
        Load audio file from local path

        Args:
            file_path (str): Path to the local audio file

        Returns:
            str: Absolute path to the audio file or None
        """
        logger.info(f"Loading local audio from path: {file_path}")
        try:
            if os.path.exists(file_path):
                abs_path = os.path.abspath(file_path)
                logger.info(f"Audio file found at: {abs_path}")
                return abs_path
            else:
                logger.error(f"Audio file not found at: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading local audio: {e}")
            return None

    @staticmethod
    def generate_fingerprint(audio_path, max_duration=2, sample_rate=22050):
        """
        Generate robust audio fingerprint

        Args:
            audio_path (str): Path to audio file
            max_duration (int): Maximum duration to analyze
            sample_rate (int): Sampling rate for processing

        Returns:
            numpy.ndarray: Normalized audio fingerprint
        """
        logger.info(f"Generating fingerprint for audio file: {audio_path}")
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=sample_rate, duration=max_duration)
            logger.info(f"Audio loaded with sample rate: {sr}, duration: {max_duration}")

            # Handle stereo to mono
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]
                logger.info("Converted stereo audio to mono")

            # Compute spectrogram
            f, t, Sxx = spectrogram(audio_data, fs=sr, nperseg=1024, noverlap=512)
            logger.info("Spectrogram computed")

            # Compress spectrogram
            compressed_spectrogram = np.mean(Sxx, axis=1)
            logger.info("Spectrogram compressed")

            # Normalize fingerprint
            normalized_fingerprint = (compressed_spectrogram - np.mean(compressed_spectrogram)) / np.std(compressed_spectrogram)
            logger.info("Fingerprint normalized")

            return normalized_fingerprint

        except Exception as e:
            logger.error(f"Fingerprint generation error: {e}")
            return None

    @staticmethod
    def calculate_similarity(query_fingerprint, reference_fingerprint):
        """
        Calculate similarity between two audio fingerprints

        Args:
            query_fingerprint (numpy.ndarray): First audio fingerprint
            reference_fingerprint (numpy.ndarray): Second audio fingerprint

        Returns:
            float: Similarity score between 0 and 1
        """
        logger.info("Calculating similarity between fingerprints")
        try:
            if query_fingerprint is None or reference_fingerprint is None:
                logger.warning("One or both fingerprints are None")
                return 0.0

            # Reshape fingerprints to ensure compatibility with cosine similarity
            query_reshaped = query_fingerprint.reshape(1, -1)
            reference_reshaped = reference_fingerprint.reshape(1, -1)

            # Calculate cosine similarity
            similarity = float(cosine_similarity(query_reshaped, reference_reshaped)[0][0])
            logger.info(f"Cosine similarity calculated: {similarity}")

            # Clip to [0, 1]
            similarity_clipped = max(0.0, min(similarity, 1.0))
            logger.info(f"Similarity clipped to range [0, 1]: {similarity_clipped}")

            return similarity_clipped

        except Exception as e:
            logger.error(f"Audio similarity calculation error: {e}")
            return 0.0
