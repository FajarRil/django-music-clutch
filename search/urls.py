from django.urls import path
from .views import SongSearchView

urlpatterns = [
    path('search/', SongSearchView.as_view(), name='song_search'),
]