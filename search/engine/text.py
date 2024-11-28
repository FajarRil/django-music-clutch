import os
import re
import logging
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class TextSimilarityEngine:
    """
    Handles text-based similarity and lyrics extraction
    """

    @staticmethod
    def clean_text(text):
        """
        Clean and normalize text input

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned and normalized text
        """
        if not text:
            return ""
        text = re.sub(r"[^\w\s]", "", str(text).lower())
        return " ".join(text.split())

    @staticmethod
    def extract_lyrics_from_audio(audio_path):
        """
        Extract text/lyrics from audio using speech recognition

        Args:
            audio_path (str): Path to audio file

        Returns:
            str: Extracted text or empty string
        """
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"Invalid audio path: {audio_path}")
            return ""

        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                return TextSimilarityEngine.clean_text(text)

        except Exception as e:
            logger.error(f"Lyrics extraction error: {e}")
            return ""

    @staticmethod
    def calculate_similarity(query_words, song_words, query_audio_path=None):
        """
        Calculate similarity between words, potentially using audio text

        Args:
            query_words (list): Words from query
            song_words (dict): Dictionary containing song words
            query_audio_path (str, optional): Path to query audio file

        Returns:
            float: Similarity score between 0 and 1
        """
        # Try audio text extraction if no query words
        if not query_words and query_audio_path:
            audio_text = TextSimilarityEngine.extract_lyrics_from_audio(query_audio_path)
            query_words = audio_text.split()

        # Prepare song words
        song_title_words = song_words.get('title_words', set())
        song_artist_words = song_words.get('artist_words', set())
        song_all_words = set(song_words.get('words', []))

        query_set = set(query_words)

        # Calculate overlaps
        title_overlap = len(query_set.intersection(song_title_words)) / max(len(query_set), len(song_title_words)) if song_title_words else 0
        artist_overlap = len(query_set.intersection(song_artist_words)) / max(len(query_set), len(song_artist_words)) if song_artist_words else 0
        word_overlap = len(query_set.intersection(song_all_words)) / max(len(query_set), len(song_all_words)) if song_all_words else 0

        # Weight different overlaps
        return max(0.0, min(0.5 * title_overlap + 0.3 * artist_overlap + 0.2 * word_overlap, 1.0))