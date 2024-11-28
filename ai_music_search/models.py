from django.db import models


class Song(models.Model):
    title = models.CharField(max_length=200)
    artist = models.CharField(max_length=200)
    cloudinary_url = models.URLField()
    lyrics = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.title} by {self.artist}"
