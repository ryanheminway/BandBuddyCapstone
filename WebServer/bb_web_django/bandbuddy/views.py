from django.shortcuts import render
from django.views.generic import TemplateView, CreateView

class HomePageView(TemplateView):
    def get(self, request, **kwargs):
       return render(request, 'index.html', context=None)

from .forms import GenreForm


def update_genre (request):
    #have it send to stage 2 here
    print(request.POST.get('genre','0'))
    
    form = GenreForm(request.POST or None) 

    context = { 
        "form":form
    }

    return render(request, "template2.html", context)