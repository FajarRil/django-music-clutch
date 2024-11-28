web: gunicorn ai_music_search.wsgi --log-file - --timeout 1800 --keep-alive 1800
web: python manage.py && gunicorn ai_music_search.wsgi