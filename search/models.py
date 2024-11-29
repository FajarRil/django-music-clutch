from django.db import models
import uuid

class Song(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200)
    artist = models.CharField(max_length=200)
    cloudinary_url = models.URLField(max_length=500)
    path = models.CharField(max_length=500)
    lyrics = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.title} by {self.artist}"