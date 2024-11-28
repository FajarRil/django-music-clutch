import logging
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


def robust_audio_load(audio_path, sample_rate=22050, max_duration=None):
    """
    Robust audio loading with fallback mechanisms

    Args:
        audio_path (str): Path to audio file
        sample_rate (int): Target sample rate
        max_duration (float, optional): Maximum duration to load

    Returns:
        tuple: Audio time series and sample rate
    """
    try:
        # Try librosa first
        y, sr = librosa.load(audio_path, sr=sample_rate, duration=max_duration)
        return y, sr
    except Exception as e:
        try:
            # Fallback to soundfile
            y, sr = sf.read(audio_path)
            if sr != sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
                sr = sample_rate

            # Truncate if max_duration specified
            if max_duration is not None:
                max_samples = int(max_duration * sr)
                y = y[:max_samples]

            return y, sr
        except Exception as load_error:
            logger.error(f"Audio loading failed for {audio_path}: {load_error}")
            return None, None


def calculate_safe_nfft(audio_length):
    """
    Calculate a safe N_FFT value based on audio length

    Args:
        audio_length (int): Length of audio signal

    Returns:
        int: Appropriate N_FFT value
    """
    return min(2048, next_power_of_two(max(audio_length // 4, 1024)))


def next_power_of_two(x):
    """Find next power of two"""
    return 2 ** int(np.ceil(np.log2(x)))


class MusicSimilarityEngine:
    """
    Handles advanced music feature extraction and similarity calculation
    """

    def __init__(self, sample_rate=22050, hop_length=512):
        """
        Initialize music similarity engine

        Args:
            sample_rate (int): Audio sampling rate
            hop_length (int): Samples between successive analysis frames
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def extract_music_features(self, audio_path):
        """
        Extract comprehensive music features from an audio file

        Args:
            audio_path (str): Path to the audio file

        Returns:
            dict: Extracted music features or None
        """
        try:
            # Robust audio loading
            y, sr = robust_audio_load(audio_path, self.sample_rate)

            if y is None or len(y) == 0:
                logger.error(f"Could not load audio file: {audio_path}")
                return None

            # Dynamically adjust n_fft
            n_fft = calculate_safe_nfft(len(y))

            # Spectral Features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=n_fft
            )[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=n_fft
            )[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=n_fft
            )[0]

            # Temporal Features
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

            # Harmonic and Percussive Components
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            # Chroma Features
            chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)

            # Mel Spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

            features = {
                "melody_features": {
                    "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                    "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                    "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                },
                "rhythm_features": {
                    "tempo": float(tempo),
                    "onset_strength_mean": float(np.mean(onset_env)),
                },
                "harmony_features": {"chroma_mean": list(np.mean(chroma_cens, axis=1))},
                "timbre_features": {
                    "harmonic_energy_ratio": float(
                        np.mean(y_harmonic**2) / np.mean(y_percussive**2)
                    ),
                    "mel_spectrogram_energy": float(np.mean(mel_spectrogram)),
                },
            }

            return features

        except Exception as e:
            logger.error(f"Music feature extraction error: {e}")
            return None

    def calculate_music_similarity(self, features1, features2):
        """
        Calculate similarity between two music feature sets

        Args:
            features1 (dict): First set of music features
            features2 (dict): Second set of music features

        Returns:
            float: Similarity score between 0 and 1
        """
        if not features1 or not features2:
            return 0.0

        try:
            logger.info("Calculating similarity between fingerprints")

            # Melody Similarity
            melody_sim = 1 - np.abs(
                features1["melody_features"]["spectral_centroid_mean"]
                - features2["melody_features"]["spectral_centroid_mean"]
            ) / max(
                features1["melody_features"]["spectral_centroid_mean"],
                features2["melody_features"]["spectral_centroid_mean"],
            )
            logger.info(f"Melody similarity calculated: {melody_sim}")

            # Rhythm Similarity
            rhythm_sim = 1 - np.abs(
                features1["rhythm_features"]["tempo"]
                - features2["rhythm_features"]["tempo"]
            ) / max(
                features1["rhythm_features"]["tempo"],
                features2["rhythm_features"]["tempo"],
            )
            logger.info(f"Rhythm similarity calculated: {rhythm_sim}")

            # Harmony Similarity
            chroma1 = np.array(features1["harmony_features"]["chroma_mean"])
            chroma2 = np.array(features2["harmony_features"]["chroma_mean"])
            harmony_sim = np.dot(chroma1, chroma2) / (
                np.linalg.norm(chroma1) * np.linalg.norm(chroma2)
            )
            logger.info(f"Harmony similarity calculated: {harmony_sim}")

            # Timbre Similarity
            timbre_sim = 1 - np.abs(
                features1["timbre_features"]["harmonic_energy_ratio"]
                - features2["timbre_features"]["harmonic_energy_ratio"]
            ) / max(
                features1["timbre_features"]["harmonic_energy_ratio"],
                features2["timbre_features"]["harmonic_energy_ratio"],
            )
            logger.info(f"Timbre similarity calculated: {timbre_sim}")

            # Weighted average of similarities
            final_sim = np.mean(
                [
                    melody_sim * 0.25,
                    rhythm_sim * 0.25,
                    harmony_sim * 0.25,
                    timbre_sim * 0.25,
                ]
            )
            final_sim = np.clip(final_sim, 0.0, 1.0)
            logger.info(f"Final similarity score: {final_sim}")

            return final_sim

        except Exception as e:
            logger.error(f"Music similarity calculation error: {e}")
            return 0.0
