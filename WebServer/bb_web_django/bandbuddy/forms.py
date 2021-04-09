#from django import forms
#from .models import MyModel

#class MyModelForm(forms.ModelForm):
#    class Meta:
#        model = MyModel
#        fields = ['genre']

import sys
from django import forms
sys.path.insert(0, '/home/brick/bandbuddy/BandBuddyCapstone/Firmware/code/stage2')
import band_buddy_msg, bb_types

BAR_CHOICES = (
        ('2', '2 BARS'),
        ('4', '4 BARS'),
)

class GenreForm(forms.Form):
 
    genre = forms.ChoiceField(choices=bb_types.GENRE_TO_ID)
    timbre = forms.ChoiceField(choices=bb_types.TIMBRE_TO_ID)
    tempo = forms.IntegerField(max_value=300, min_value=30)
    temperature = forms.DecimalField(max_value=1, min_value=0)
    bars = forms.ChoiceField(choices=bb_types.BARS_TO_VALUE)
    drums = forms.BooleanField(required=False)
    guitar = forms.BooleanField(required=False)
    velocity = forms.DecimalField(max_value=1, min_value=0)

    def __init__(self, *args,**kwargs):
        self.genre_val = kwargs.pop('genre')
        self.timbre_val = kwargs.pop('timbre')
        self.tempo_val = kwargs.pop('tempo')
        self.temperature_val = kwargs.pop('temperature')
        self.bars_val = kwargs.pop('bars')
        self.drums_val = kwargs.pop('drums')
        self.guitar_val = kwargs.pop('guitar')
        self.velocity_val = kwargs.pop('velocity')
        super(GenreForm, self).__init__(*args, **kwargs)
        self.fields['genre'].initial = self.genre_val
        self.fields['timbre'].initial= self.timbre_val
        self.fields['tempo'].initial = self.tempo_val
        self.fields['temperature'].initial = self.temperature_val
        self.fields['bars'].initial = self.bars_val
        self.fields['drums'].initial = self.drums_val
        self.fields['guitar'].initial = self.guitar_val
        self.fields['velocity'].initial = self.velocity_val
