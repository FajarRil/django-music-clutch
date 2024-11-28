import os
import tempfile
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from search.utils import AIMusicalFingerprintEngine

logger = logging.getLogger(__name__)

# Sample songs metadata (replace with database or external source)
SONGS_METADATA = [
    {
        "id": 1,
        "title": "Golden Hour",
        "artist": "JVKE",
        "cloudinary_url": "https://res.cloudinary.com/dutlw7bko/video/upload/v1732767187/music-clutch/golden_hour_jao7gd.mp3",
        "lyrics": "It was just two lovers Sittin in the car, listening to Blonde Fallin for each other Pink and orange skies, feelin super childish No Donald Glover Missed call from my mother Like, Where you at tonight? Got no alibi I was all alone with the love of my life Shes got glitter for skin My radiant beam in the night I dont need no light to see you Shine Its your golden hour (oh) You slow down time In your golden hour (oh) We were just two lovers Feet up on the dash, drivin nowhere fast Burnin through the summer Radio on blast, make the moment last She got solar power Minutes feel like hours She knew she was the baddest, can you even imagine Fallin like I did? For the love of my life Shes got glow on her face A glorious look in her eyes My angel of light I was all alone with the love of my life Shes got glitter for skin My radiant beam in the night I dont need no light to see you Shine Its your golden hour (oh) You slow down time In your golden hour (oh)",
    },
    {
        "id": 2,
        "title": "Sadrah",
        "artist": "For Revenge",
        "cloudinary_url": "https://res.cloudinary.com/dutlw7bko/video/upload/v1732767181/music-clutch/sadrah_jkytcl.mp3",
        "lyrics": "Aku yang dipaksa menyerah Jadi yang paling salah Sementara kau dengannya Aku yang kini terbuang Kaujadikan pecundang Sementara kau bersulang Sudahlah, kali ini aku kalah Kehilangan mahkota Kau dan dia pemenangnya Berakhir, tak usah khawatir Tak mengapa (tak mengapa) Ku hanya harus terima (aku kan menerima) Tersingkir, tak usah permisi Tak mengapa (tak mengapa) Bahagialah bersamanya Sudahlah, kali ini aku kalah Kehilangan mahkota Kau dan dia pemenangnya Sudahlah, kali ini ku berserah Kehilangan segalanya Kau dan dia pemenangnya Lelah harus bijaksana Saat kita yang terluka Lelah harus bijaksana Saat kita yang terluka Sudahlah, kali ini aku kalah Kehilangan mahkota Kau dan dia pemenangnya Sudahlah, kali ini ku berserah Kehilangan segalanya Kau dan dia pemenangnya Sepatutnya kaurayakan",
    },
    {
        "id": 3,
        "title": "Die With A Smile",
        "artist": "Lady Gaga & Bruno Mars",
        "cloudinary_url": "https://res.cloudinary.com/dutlw7bko/video/upload/v1732767181/music-clutch/die_with_a_smile_hg9tbz.mp3",
        "lyrics": "I, I just woke up from a dream Where you and I had to say goodbye And I dont know what it all means But since I survived, I realized Wherever you go, thats where Ill follow Nobodys promised tomorrow So Ima love you every night like its the last night Like its the last night If the world was ending Id wanna be next to you If the party was over And our time on Earth was through Id wanna hold you just for a while And die with a smile If the world was ending Id wanna be next to you Ooh, lost, lost in the words that we scream I dont even wanna do this anymore Cause you already know what you mean to me And our loves the only one worth fighting for Wherever you go, thats where Ill follow Nobodys promised tomorrow So Ima love you every night like its the last night Like its the last night If the world was ending Id wanna be next to you If the party was over And our time on Earth was through Id wanna hold you just for a while And die with a smile If the world was ending Id wanna be next to you Right next to you Next to you Right next to you Oh-oh If the world was ending Id wanna be next to you If the party was over And our time on Earth was through Id wanna hold you just for a while And die with a smile If the world was ending Id wanna be next to you If the world was ending Id wanna be next to you Id wanna be next to you",
    },
]


class SongSearchView(APIView):
    def post(self, request):
        try:
            audio_file = request.FILES.get('audio')
            query_text = request.data.get('text', '')

            if not audio_file:
                return Response(
                    {"message": "No audio file provided", "status": "error"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate file size
            if audio_file.size > 10 * 1024 * 1024:  # 10MB limit
                return Response(
                    {"message": "File too large. Max 10MB.", "status": "error"},
                    status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                )

            # Save uploaded file temporarily
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, audio_file.name)
            
            with open(temp_path, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)

            # Initialize music engine
            music_engine = AIMusicalFingerprintEngine(SONGS_METADATA)

            # Match song
            matches = music_engine.match_song(temp_path, query_text)

            # Clean up temporary file
            os.unlink(temp_path)
            os.rmdir(temp_dir)

            return Response(matches, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Search API error: {str(e)}")
            return Response(
                {"message": "Internal server error", "status": "error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )