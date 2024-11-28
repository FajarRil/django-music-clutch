import os
import re
import scipy
import logging
import requests
import tempfile
import numpy as np
import soundfile as sf
import speech_recognition as sr
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class AIMusicalFingerprintEngine:
    def __init__(self, songs):
        self.songs_metadata = songs
        self.text_vectorizer = TfidfVectorizer(stop_words="english")
        self.audio_fingerprints = []

        self.prepare_text_features()
        self.generate_audio_fingerprints()

    def clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r"[^\w\s]", "", str(text).lower())
        return " ".join(text.split())

    def prepare_text_features(self):
        text_corpus = []
        self.word_level_features = {}

        for song in self.songs_metadata:
            song_id = str(song["id"])

            full_text = self.clean_text(
                f"{song.get('title', '')} {song.get('artist', '')} {song.get('lyrics', '')}"
            )

            self.word_level_features[song_id] = {
                "words": full_text.split(),
                "title_words": set(self.clean_text(song.get("title", "")).split()),
                "artist_words": set(self.clean_text(song.get("artist", "")).split()),
            }

            text_corpus.append(full_text)

        self.text_vectorizer = TfidfVectorizer(stop_words="english")
        self.text_matrix = self.text_vectorizer.fit_transform(text_corpus)

    def download_audio(self, url):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            try:
                response = requests.get(url, stream=True, timeout=10)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                return temp_file.name
            except requests.exceptions.RequestException as e:
                logger.error(f"Audio download error: {e}")
                return None

    def generate_audio_fingerprint(self, audio_path, max_duration=10):
        try:
            # Convert webm to wav format that soundfile can read
            if audio_path.endswith(".webm"):
                audio = AudioSegment.from_file(audio_path, format="webm")
                wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
                audio.export(wav_path, format="wav")
                audio_data, sample_rate = sf.read(wav_path)
                os.remove(wav_path)  # Clean up temporary wav file
            else:
                audio_data, sample_rate = sf.read(audio_path)

            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]

            if len(audio_data) > max_duration * sample_rate:
                audio_data = audio_data[: max_duration * sample_rate]

            _, _, spectrogram_data = scipy.signal.spectrogram(
                audio_data, fs=sample_rate, nperseg=1024, noverlap=512, mode="magnitude"
            )

            if spectrogram_data.size == 0:
                logger.error(f"Empty spectrogram for {audio_path}")
                return None

            compressed_spectrogram = np.mean(spectrogram_data, axis=1)
            normalized_fingerprint = (
                compressed_spectrogram - np.mean(compressed_spectrogram)
            ) / np.std(compressed_spectrogram)

            return normalized_fingerprint
        except Exception as e:
            logger.error(f"Fingerprint generation error: {e}")
            return None

    def generate_audio_fingerprints(self):
        for song in self.songs_metadata:
            file_url = song.get("cloudinary_url", "")
            if file_url:
                try:
                    temp_file = self.download_audio(file_url)
                    if temp_file:
                        fingerprint = self.generate_audio_fingerprint(temp_file)
                        if fingerprint is not None:
                            self.audio_fingerprints.append(fingerprint)
                        os.unlink(temp_file)
                except Exception as e:
                    logger.error(f"Fingerprint generation error: {e}")

    def extract_lyrics_from_audio(self, audio_path):
        """
        Extract lyrics/text from audio using Google Speech Recognition
        """
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"Invalid audio path: {audio_path}")
            return ""

        logger.info(f"Starting lyrics extraction from: {audio_path}")
        try:

            # Initialize recognizer
            recognizer = sr.Recognizer()

            # Convert audio file to AudioFile
            with sr.AudioFile(audio_path) as source:
                # Read audio data
                audio_data = recognizer.record(source)
                
                # Use Google Speech Recognition
                logger.info("Transcribing audio with Google Speech Recognition")
                text = recognizer.recognize_google(audio_data)
                logger.info(f"Raw extracted text: {text}")

                cleaned_text = self.clean_text(text)
                logger.info(f"Cleaned extracted text: {cleaned_text}")
                return cleaned_text

        except Exception as e:
            logger.error(f"Error extracting lyrics from audio: {e}")
            return ""

    def calculate_word_similarity(self, query_words, song_words, query_audio_path=None):
        """
        Calculate similarity between input audio lyrics and song lyrics

        Args:
            query_words (list): Words from the query (not used in this version)
            song_words (dict): Dictionary containing song word features
            query_audio_path (str, optional): Path to audio file for lyrics extraction

        Returns:
            float: Similarity score between 0 and 1
        """
        logger.info("Starting word similarity calculation")
        logger.debug(f"Query words: {query_words}")
        logger.debug(f"Song words: {song_words}")

        if not query_audio_path:
            logger.warning("No audio path provided")
            return 0.0

        # Extract lyrics from input audio
        logger.info(f"Extracting text from audio file: {query_audio_path}")
        audio_text = self.extract_lyrics_from_audio(query_audio_path).split()
        logger.info(f"Extracted {len(audio_text)} words from audio")
        logger.debug(f"Extracted audio text: {audio_text}")

        if not audio_text:
            logger.warning("No text extracted from audio")
            return 0.0

        # Get song lyrics
        song_lyrics = set(song_words.get("words", []))
        logger.info(f"Song lyrics contain {len(song_lyrics)} words")
        if not song_lyrics:
            logger.warning("No lyrics found in song")
            return 0.0

        # Calculate similarity between audio lyrics and song lyrics
        audio_word_set = set(audio_text)
        common_words = audio_word_set.intersection(song_lyrics)
        logger.info(f"Found {len(common_words)} common words")
        logger.debug(f"Common words: {common_words}")
        
        similarity = len(common_words) / max(len(audio_word_set), len(song_lyrics))
        logger.info(f"Final lyrics similarity score: {similarity}")
        
        return min(max(similarity, 0.0), 1.0)

    def match_song(self, query_audio_path, query_text=None):
        try:
            query_fingerprint = self.generate_audio_fingerprint(query_audio_path)

            if query_fingerprint is None:
                return {
                    "message": "No valid audio fingerprints generated",
                    "audio_similarity": 0.0,
                    "word_similarity": 0.0,
                    "hybrid_score": 0.0,
                    "title": "",
                    "artist": "",
                    "cloudinary_url": "",
                    "lyrics": ""
                }

            valid_fingerprints = [(idx, fp) for idx, fp in enumerate(self.audio_fingerprints) if fp is not None]

            if not valid_fingerprints:
                return {
                    "message": "No valid audio fingerprints in dataset",
                    "audio_similarity": 0.0,
                    "word_similarity": 0.0,
                    "hybrid_score": 0.0,
                    "title": "",
                    "artist": "",
                    "cloudinary_url": "",
                    "lyrics": ""
                }

            matches = []
            query_words = self.clean_text(query_text or os.path.basename(query_audio_path)).split()

            for idx, fingerprint in valid_fingerprints:
                audio_sim = float(cosine_similarity(
                    query_fingerprint.reshape(1, -1), 
                    fingerprint.reshape(1, -1)
                )[0][0])

                song_id = str(self.songs_metadata[idx]["id"])
                word_sim = float(self.calculate_word_similarity(
                    query_words, 
                    self.word_level_features[song_id], 
                    query_audio_path
                ))

                hybrid_score = (audio_sim + word_sim) / 2

                matches.append({
                    "title": self.songs_metadata[idx].get("title", ""),
                    "artist": self.songs_metadata[idx].get("artist", ""),
                    "audio_similarity": audio_sim,
                    "word_similarity": word_sim,
                    "hybrid_score": hybrid_score,
                    "cloudinary_url": self.songs_metadata[idx].get("cloudinary_url", ""),
                    "lyrics": self.songs_metadata[idx].get("lyrics", "")
                })

            text_matches = [m for m in matches if m["word_similarity"] > 0]
            if text_matches:
                best_match = max(text_matches, key=lambda x: x["word_similarity"])
                best_match["message"] = "Match found based on lyrics"
                return best_match
            
            if matches:
                best_match = max(matches, key=lambda x: x["audio_similarity"])
                best_match["message"] = "Match found based on audio similarity"
                return best_match

            return {
                "message": "No matches found",
                "audio_similarity": 0.0,
                "word_similarity": 0.0,
                "hybrid_score": 0.0,
                "title": "",
                "artist": "",
                "cloudinary_url": "",
                "lyrics": ""
            }

        except Exception as e:
            logger.error(f"Song matching error: {e}")
            return {
                "message": f"Matching error: {str(e)}",
                "audio_similarity": 0.0,
                "word_similarity": 0.0,
                "hybrid_score": 0.0,
                "title": "",
                "artist": "",
                "cloudinary_url": "",
                "lyrics": ""
            }