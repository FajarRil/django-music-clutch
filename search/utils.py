import os
import logging
import numpy as np
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer

from search.engine.audio import AudioSimilarityEngine
from search.engine.music import MusicSimilarityEngine
from search.engine.text import TextSimilarityEngine  # Ensure this import is correct


class AIMusicalFingerprintEngine:
    """
    A comprehensive engine for music matching and similarity analysis

    This class provides advanced capabilities for:
    - Audio fingerprinting
    - Music feature extraction
    - Text and lyrics similarity
    - Hybrid song matching
    """

    def __init__(self, songs, max_duration=10, sample_rate=22050):
        """
        Initialize the musical fingerprint engine

        Args:
            songs (list): List of song metadata dictionaries
            max_duration (int): Maximum audio duration to process in seconds
            sample_rate (int): Audio sampling rate for feature extraction
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Configuration parameters
        self.max_duration = max_duration
        self.sample_rate = sample_rate

        # Song metadata and feature storage
        self.songs_metadata = songs
        self.text_vectorizer = TfidfVectorizer(stop_words="english")
        self.audio_fingerprints = []
        self.music_features = []
        self.word_level_features = {}

        # Prepare features
        self.logger.info("Preparing features for all songs")
        self._prepare_features()

    def _prepare_features(self):
        """
        Prepare text and audio features for all songs
        """
        try:
            self.logger.info("Preparing text features")
            self._prepare_text_features()
            self.logger.info("Generating audio fingerprints")
            self._generate_audio_fingerprints()
        except Exception as e:
            self.logger.error(f"Feature preparation error: {e}")

    def _prepare_text_features(self):
        """
        Prepare text-based features for songs
        """
        text_corpus = []
        self.word_level_features = {}

        for song in self.songs_metadata:
            song_id = str(song.get("id", ""))
            full_text = TextSimilarityEngine.clean_text(
                f"{song.get('title', '')} {song.get('artist', '')} {song.get('lyrics', '')}"
            )

            self.word_level_features[song_id] = {
                "words": set(full_text.split()),
                "title_words": set(
                    TextSimilarityEngine.clean_text(song.get("title", "")).split()
                ),
                "artist_words": set(
                    TextSimilarityEngine.clean_text(song.get("artist", "")).split()
                ),
            }

            text_corpus.append(full_text)

        self.text_matrix = self.text_vectorizer.fit_transform(text_corpus)
        self.logger.info("Text features prepared")

    def _generate_audio_fingerprints(self):
        """
        Generate audio fingerprints and music features for all songs
        """
        for song in self.songs_metadata:
            file_url = song.get("path", "")
            if file_url:
                try:
                    self.logger.info(f"Processing audio file from URL: {file_url}")

                    # Check if file is from music folder or a local file
                    is_local_music_file = (
                        os.path.exists(file_url) and "music/" in file_url
                    )

                    # If it's a local music file, use it directly
                    if is_local_music_file:
                        temp_file = file_url
                    else:
                        # For other sources, use _load_local_audio to handle temporary files
                        temp_file = self._load_local_audio(file_url)

                    if temp_file:
                        fingerprint = AudioSimilarityEngine.generate_fingerprint(
                            temp_file
                        )
                        music_feature = MusicSimilarityEngine().extract_music_features(
                            temp_file
                        )

                        if fingerprint is not None and music_feature is not None:
                            self.audio_fingerprints.append(fingerprint)
                            self.music_features.append(music_feature)
                            self.logger.info(
                                f"Generated fingerprint and features for {file_url}"
                            )

                        # Only unlink if not a local music file
                        if not is_local_music_file and os.path.exists(temp_file):
                            os.unlink(temp_file)

                except Exception as e:
                    self.logger.error(f"Fingerprint generation error: {e}")

    def _load_local_audio(self, file_path):
        """
        Load audio file from local or remote path, creating a temporary file if needed

        Args:
            file_path (str): Path to the audio file

        Returns:
            str: Absolute path to the audio file or temporary file
        """
        self.logger.info(f"Loading audio from path: {file_path}")
        try:
            # If file exists and is in music folder, return direct path
            if os.path.exists(file_path) and "music/" in file_path:
                return file_path

            # For remote or other URLs, create a temporary file
            if os.path.exists(file_path):
                abs_path = os.path.abspath(file_path)
                self.logger.info(f"Audio file found at: {abs_path}")
                return abs_path

            # Placeholder for potential remote file download logic
            self.logger.error(f"Audio file not found: {file_path}")
            return None

        except Exception as e:
            self.logger.error(f"Error loading audio: {e}")
            return None

    def match_song(self, query_audio_path, query_text=None):
        """
        Match input audio against database of songs

        Args:
            query_audio_path (str): Path to the query audio file
            query_text (str, optional): Additional text for matching

        Returns:
            dict: Matching song details and similarity scores
        """
        try:
            # Determine if a temporary file needs to be created
            is_temp_file = False
            temp_file = query_audio_path

            # Create a temporary file if the input is not a local file
            if not os.path.exists(query_audio_path):
                temp_dir = tempfile.mkdtemp()
                temp_file = os.path.join(temp_dir, os.path.basename(query_audio_path))
                with open(temp_file, "wb+") as destination:
                    with open(query_audio_path, "rb") as source:
                        destination.write(source.read())
                is_temp_file = True

            self.logger.info(f"Matching song for query audio: {temp_file}")
            query_fingerprint = AudioSimilarityEngine.generate_fingerprint(temp_file)
            query_music_features = MusicSimilarityEngine().extract_music_features(
                temp_file
            )

            # Clean up temporary file if created
            if is_temp_file:
                os.unlink(temp_file)
                os.rmdir(os.path.dirname(temp_file))

            if query_fingerprint is None or query_music_features is None:
                return self._no_match_result("Invalid query audio features")

            matches = []
            query_text = query_text or os.path.basename(temp_file)
            query_words = TextSimilarityEngine.clean_text(query_text).split()
            music_similarity_engine = MusicSimilarityEngine()

            for idx, (fingerprint, music_feature) in enumerate(
                zip(self.audio_fingerprints, self.music_features)
            ):
                audio_sim = AudioSimilarityEngine.calculate_similarity(
                    query_fingerprint, fingerprint
                )
                music_sim = music_similarity_engine.calculate_music_similarity(
                    query_music_features, music_feature
                )
                song_id = str(self.songs_metadata[idx].get("id", ""))

                # Use a dictionary with correct keys matching TextSimilarityEngine.calculate_similarity expectations
                word_sim = TextSimilarityEngine.calculate_similarity(
                    query_words,
                    {
                        "title_words": self.word_level_features.get(song_id, {}).get(
                            "title_words", set()
                        ),
                        "artist_words": self.word_level_features.get(song_id, {}).get(
                            "artist_words", set()
                        ),
                        "words": self.word_level_features.get(song_id, {}).get(
                            "words", set()
                        ),
                    },
                    query_audio_path,  # Pass audio path in case text recognition is needed
                )

                hybrid_score = np.mean([audio_sim, music_sim, word_sim])

                matches.append(
                    {
                        "title": self.songs_metadata[idx].get("title", ""),
                        "artist": self.songs_metadata[idx].get("artist", ""),
                        "audio_similarity": audio_sim,
                        "music_similarity": music_sim,
                        "word_similarity": word_sim,
                        "hybrid_score": hybrid_score,
                        "cloudinary_url": self.songs_metadata[idx].get(
                            "cloudinary_url", ""
                        ),
                        "lyrics": self.songs_metadata[idx].get("lyrics", ""),
                    }
                )

            if not matches:
                return self._no_match_result("No matches found")

            # Filtering logic modified to handle potential edge cases
            text_matches = [m for m in matches if m["word_similarity"] > 0]
            if text_matches:
                best_match = max(text_matches, key=lambda x: x["hybrid_score"])
                best_match["message"] = "Match found based on lyrics and music features"
                return best_match

            best_match = max(matches, key=lambda x: x["hybrid_score"])
            best_match["message"] = "Match found based on audio and music similarity"
            self.logger.info(f"Match found: {best_match}")
            return best_match

        except Exception as e:
            return self._no_match_result(f"Matching error: {str(e)}")

    def _no_match_result(self, message):
        """
        Return a standardized no match result

        Args:
            message (str): Message describing the reason for no match

        Returns:
            dict: No match result with message
        """
        self.logger.info(f"No match result: {message}")
        return {
            "title": "",
            "artist": "",
            "audio_similarity": 0.0,
            "music_similarity": 0.0,
            "word_similarity": 0.0,
            "hybrid_score": 0.0,
            "cloudinary_url": "",
            "lyrics": "",
            "message": message,
        }
