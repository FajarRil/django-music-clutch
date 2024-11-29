import os
import re
import logging
import numpy as np
from typing import List, Dict, Optional
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

logger = logging.getLogger(__name__)


class TextSimilarityEngine:
    """
    Advanced text similarity engine with multi-level text processing

    Features:
    - Advanced text cleaning
    - Audio text extraction
    - Lemmatization
    - Stop word removal
    - Weighted similarity calculation
    - Multiple similarity metrics
    """

    # Precompute stop words for efficiency
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()

    @classmethod
    def clean_text(cls, text: str, advanced_cleaning: bool = True) -> str:
        """
        Advanced text cleaning method

        Args:
            text (str): Input text to clean
            advanced_cleaning (bool): Enable advanced cleaning techniques

        Returns:
            str: Cleaned and normalized text
        """
        if not text:
            return ""

        # Convert to lowercase and remove special characters
        cleaned_text = re.sub(r"[^\w\s]", "", str(text).lower())

        if advanced_cleaning:
            # Tokenize
            tokens = cleaned_text.split()

            # Remove stop words and lemmatize
            processed_tokens = [
                cls.LEMMATIZER.lemmatize(token)
                for token in tokens
                if token not in cls.STOP_WORDS
            ]

            return " ".join(processed_tokens)

        return " ".join(cleaned_text.split())

    @classmethod
    def extract_lyrics_from_audio(
        cls, audio_path: str, language: str = "en-US", timeout: int = 10
    ) -> str:
        """
        Advanced lyrics extraction with multiple recognition strategies

        Args:
            audio_path (str): Path to audio file
            language (str): Language for speech recognition
            timeout (int): Maximum time for recognition attempt

        Returns:
            str: Extracted and cleaned text
        """
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"Invalid audio path: {audio_path}")
            return ""

        try:
            recognizer = sr.Recognizer()

            # Adjust for ambient noise
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                recognizer.adjust_for_ambient_noise(source, duration=1)

            # Multiple recognition attempts with different parameters
            recognition_strategies = [
                lambda: recognizer.recognize_google(audio_data, language=language),
                lambda: recognizer.recognize_sphinx(audio_data),
                lambda: recognizer.recognize_whisper(audio_data),
                lambda: recognizer.recognize_wit(audio_data),
                lambda: recognizer.recognize_azure(audio_data),
                lambda: recognizer.recognize_ibm(audio_data),
            ]

            for strategy in recognition_strategies:
                try:
                    text = strategy()
                    if text:
                        return cls.clean_text(text)
                except Exception as e:
                    logger.debug(f"Recognition strategy failed: {e}")

            return ""

        except Exception as e:
            logger.error(f"Comprehensive lyrics extraction error: {e}")
            return ""

    @classmethod
    def calculate_similarity(
        cls,
        query_words: List[str],
        song_words: Dict[str, set],
        query_audio_path: Optional[str] = None,
    ) -> float:
        """
        Advanced similarity calculation with multiple scoring mechanisms

        Args:
            query_words (List[str]): Words from query
            song_words (Dict[str, set]): Dictionary containing song words
            query_audio_path (Optional[str]): Path to query audio file

        Returns:
            float: Comprehensive similarity score between 0 and 1
        """
        logger.debug("Starting similarity calculation")

        # Extract words from audio if no query words
        if not query_words and query_audio_path:
            logger.info(f"Extracting lyrics from audio: {query_audio_path}")
            audio_text = cls.extract_lyrics_from_audio(query_audio_path)
            logger.debug(f"Extracted lyrics: {audio_text}")
            query_words = cls.clean_text(audio_text).split()

        if not query_words:
            logger.warning("No query words provided for similarity calculation")
            return 0.0

        # Prepare song words with advanced processing
        logger.debug("Preparing song words for comparison")
        song_title_words = song_words.get("title_words", set())
        song_artist_words = song_words.get("artist_words", set())
        song_all_words = set(song_words.get("words", []))

        # Convert query words to set for efficient operations
        query_set = set(song_all_words)
        logger.debug(f"Query words: {query_set}")

        # Advanced overlap calculations with weighted semantic importance
        def calculate_weighted_overlap(query_set, target_set, weight=1.0):
            """Calculate weighted set overlap with semantic scoring"""
            if not target_set:
                return 0.0

            # Jaccard similarity with weighted scoring
            intersection = len(query_set.intersection(target_set))
            union = len(query_set.union(target_set))

            # Cosine-like similarity with weight
            similarity = weight * (intersection / (union + 1e-10))
            logger.debug(f"Overlap score: {similarity} (weight: {weight})")
            return similarity

        # Comprehensive similarity scoring
        logger.debug("Calculating weighted overlaps")
        title_overlap = calculate_weighted_overlap(
            query_set, song_title_words, weight=0.4
        )
        artist_overlap = calculate_weighted_overlap(
            query_set, song_artist_words, weight=0.3
        )
        word_overlap = calculate_weighted_overlap(query_set, song_all_words, weight=0.3)

        # Advanced scoring with dynamic weighting
        total_similarity = title_overlap + artist_overlap + word_overlap
        logger.debug(f"Final similarity score: {total_similarity}")

        # Normalization and clamping
        return max(0.0, min(total_similarity, 1.0))

    @staticmethod
    def advanced_text_vectorization(texts: List[str]) -> np.ndarray:
        """
        Advanced text vectorization using TF-IDF with enhanced preprocessing

        Args:
            texts (List[str]): List of texts to vectorize

        Returns:
            np.ndarray: Vectorized representation of texts
        """
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),  # Consider both unigrams and bigrams
        )
        return vectorizer.fit_transform(texts)