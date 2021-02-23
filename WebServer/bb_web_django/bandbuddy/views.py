import sys
sys.path.insert(0, '/home/brick/bandbuddy/BandBuddyCapstone/Firmware/code/stage2')
import band_buddy_msg
from django.shortcuts import render
from django.views.generic import TemplateView, CreateView

class HomePageView(TemplateView):
    def get(self, request, **kwargs):
       return render(request, 'index.html', context=None)

from .forms import GenreForm

#coordinate with ryan
genre_dict= {'rock': 0, "jazz": 1, "funk": 2, "pop": 3, "classical": 4}


def update_genre (request):
    #have it send to stage 2 here
    print(request.POST.get('genre','0'))
    host = '127.0.0.1'
    port = 8080
    socket_fd = band_buddy_msg.connect_and_register(host, port)
    
    #update to send number when avaliable
    band_buddy_msg.send_midi_data(socket_fd,bytearray([0,1,2,3]),4)
    
    form = GenreForm(request.POST or None) 

    context = { 
        "form":form
    }

    return render(request, "template2.html", context)