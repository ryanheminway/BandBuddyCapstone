#from django import forms
#from .models import MyModel

#class MyModelForm(forms.ModelForm):
#    class Meta:
#        model = MyModel
#        fields = ['genre']

from django import forms
from .models import Genre
from django.forms import ModelChoiceField

class GenreForm(forms.ModelForm):
 
    class Meta:
        model = Genre
        fields = ["genre"]