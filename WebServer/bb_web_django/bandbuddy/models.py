from django.db import models

# Create your models here.
GENRE_CHOICES = (
    ('rock','ROCK'),
    ('jazz', 'JAZZ'),
    ('funk','FUNK'),
    ('pop','POP'),
    ('classical','CLASSICAL'),
)

#have default check what is currently selected here
class Genre(models.Model):
    genre = models.CharField(max_length=10, choices=GENRE_CHOICES, default='rock')