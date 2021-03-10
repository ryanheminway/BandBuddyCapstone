#from django import forms
#from .models import MyModel

#class MyModelForm(forms.ModelForm):
#    class Meta:
#        model = MyModel
#        fields = ['genre']

from django import forms

GENRE_CHOICES = (
    ('rock','ROCK'),
    ('jazz', 'JAZZ'),
    ('funk','FUNK'),
    ('pop','POP'),
    ('classical','CLASSICAL'),
)

TIMBRE_CHOICES = (
    ('timbre1', 'TIMBRE1'),
    ('timbre2', 'TIMBRE2'),
    ('timbre3', 'TIMBRE3'),
    ('timbre4', 'TIMBRE4'),
    ('timbre5', 'TIMBRE5'),
)

class GenreForm(forms.Form):
 
    genre = forms.ChoiceField(choices=GENRE_CHOICES)
    timbre = forms.ChoiceField(choices=TIMBRE_CHOICES)
    tempo = forms.IntegerField(max_value=200, min_value=30)
    temperature = forms.DecimalField(max_value=1, min_value=0)
    drums = forms.BooleanField(required=False, initial=True)
    guitar = forms.BooleanField(required=False, initial=True)
