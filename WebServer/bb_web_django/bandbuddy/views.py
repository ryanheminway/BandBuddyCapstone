import sys
sys.path.insert(0, '/home/patch/bandbuddy/BandBuddyCapstone/Firmware/code/stage2')
import band_buddy_msg
from django.shortcuts import render
from django.views.generic import TemplateView, CreateView

class HomePageView(TemplateView):
    def get(self, request, **kwargs):
       return render(request, 'index.html', context=None)

from .forms import GenreForm

#coordinate with ryan
genre_dict= {'rock': 0, "jazz": 1, "funk": 2, "pop": 3, "classical": 4}
timbre_dict= {'timbre1':0, 'timbre2':1, 'timbre3':2, 'timbre4':3, 'timbre5':4}

def update_genre (request):
    #have it send to stage 2 here
    genre = request.POST.get('genre','0')
    timbre = request.POST.get('timbre','0')
    tempo = request.POST.get('tempo','0')
    temperature = request.POST.get('temperature','0')
    drums = False
    guitar = False
    if request.POST.get('drums','0') == 'on':
        drums = True
    if request.POST.get('guitar','0') == 'on':
        guitar = True

    host = '127.0.0.1'
    port = 8080
    socket_fd = band_buddy_msg.connect_and_register(host, port, band_buddy_msg.WEB_SERVER_STAGE)  

    if genre == '0':
        band_buddy_msg.request_params(socket_fd,band_buddy_msg.WEB_SERVER_STAGE,band_buddy_msg.STAGE2)
        cmd, message = band_buddy_msg.recv_msg(socket_fd)
        print(message.Genre())
        print(message.Timbre())
        print(message.Tempo())
        print(message.Temperature())

    if genre != '0':
          
        print(genre)
        print(timbre)
        print(tempo)
        print(temperature)
        print(drums)
        print(guitar)
        band_buddy_msg.send_webserver_data(socket_fd, genre_dict[genre], timbre_dict[timbre], int(tempo), float(temperature), drums, guitar, band_buddy_msg.STAGE2, band_buddy_msg.WEB_SERVER_STAGE)

    form = GenreForm(request.POST or None,genre='jazz',timbre='timbre3',tempo=77,temperature=0.77,drums=True,guitar=False) 
    
    context = { 
        "form":form
    }

    return render(request, "template2.html", context)
