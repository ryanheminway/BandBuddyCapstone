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

BAR_CHOICES = (
        (2, '2 BARS'),
        (4, '4 BARS'),
)

class GenreForm(forms.Form):
 
    genre = forms.ChoiceField(choices=GENRE_CHOICES)
    timbre = forms.ChoiceField(choices=TIMBRE_CHOICES)
    tempo = forms.IntegerField(max_value=200, min_value=30)
    temperature = forms.DecimalField(max_value=1, min_value=0)
    bars = forms.ChoiceField(choices=BAR_CHOICES)
    drums = forms.BooleanField(required=False)
    guitar = forms.BooleanField(required=False)

    def __init__(self, *args,**kwargs):
        self.genre_val = kwargs.pop('genre')
        self.timbre_val = kwargs.pop('timbre')
        self.tempo_val = kwargs.pop('tempo')
        self.temperature_val = kwargs.pop('temperature')
        self.bars_val = kwargs.pop('bars')
        self.drums_val = kwargs.pop('drums')
        self.guitar_val = kwargs.pop('guitar')
        super(GenreForm, self).__init__(*args, **kwargs)
        self.fields['genre'].initial = self.genre_val
        self.fields['timbre'].initial= self.timbre_val
        self.fields['tempo'].initial = self.tempo_val
        self.fields['temperature'].initial = self.temperature_val
        self.fields['bars'].initial = self.bars_val
        self.fields['drums'].initial = self.drums_val
        self.fields['guitar'].initial = self.guitar_val
