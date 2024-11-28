# ai_music_search/wsgi.py

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_music_search.settings')

try:
	application = get_wsgi_application()
except Exception as e:
	raise RuntimeError("WSGI application 'ai_music_search.wsgi.application' could not be loaded; Error importing module.") from e

application = get_wsgi_application()