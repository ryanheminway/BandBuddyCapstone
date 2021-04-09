import sys
sys.path.insert(0, '/home/patch/BandBuddyCapstone/Firmware/code/stage2')
import band_buddy_msg, bb_types
import mimetypes
import soundfile as sf
from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import TemplateView, CreateView

class HomePageView(TemplateView):
    def get(self, request, **kwargs):
       return render(request, 'index.html', context=None)

from .forms import GenreForm

def download_file(request, fl_path, filename):
    fl = open(fl_path, 'rb')
    mime_type, _ = mimetypes.guess_type(fl_path)
    response = HttpResponse(fl, content_type=mime_type)
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    return response

def download_combined(request):
    fl_path = '/home/patch/wavs/sync.wav'
    filename = 'sync.wav'
    return download_file(request, fl_path, filename)

def download_drums(request):
    fl_path = '/home/patch/wavs/drums.wav'
    filename = 'drums.wav'
    return download_file(request, fl_path, filename)

def download_input(request):
    fl_path = '/home/patch/wavs/input.wav'
    filename = 'input.wav'
    return download_file(request, fl_path, filename)

def update_genre (request):
    #have it send to stage 2 here
    genre = request.POST.get('genre','0')
    timbre = request.POST.get('timbre','0')
    tempo = request.POST.get('tempo','0')
    temperature = request.POST.get('temperature','0')
    velocity = request.POST.get('velocity','0')
    bars = request.POST.get('bars','0')
    drums = False
    guitar = False
    if 'drums' in request.POST:
        drums = True
    if 'guitar' in request.POST:
        guitar = True

    host = '127.0.0.1'
    port = 8080
    socket_fd = band_buddy_msg.connect_and_register(host, port, band_buddy_msg.WEB_SERVER_STAGE)  

    if genre == '0':
        band_buddy_msg.request_params(socket_fd,band_buddy_msg.WEB_SERVER_STAGE,band_buddy_msg.STAGE2)
        cmd, message2 = band_buddy_msg.recv_msg(socket_fd)
        band_buddy_msg.request_params(socket_fd,band_buddy_msg.WEB_SERVER_STAGE,band_buddy_msg.STAGE3)
        cmd, message3 = band_buddy_msg.recv_msg(socket_fd)
        socket_fd.close()
        
        form = GenreForm(request.POST or None, genre=int(message2.Genre()),timbre=int(message2.Timbre()),tempo=message2.Tempo(),temperature=message2.Temperature(),drums=bool(message3.Drums()),guitar=bool(message3.Guitar()), bars=int(message2.Bars()), velocity=message2.Velocity()) 

    if genre != '0':
          
        print(genre)
        print(timbre)
        print(tempo)
        print(temperature)
        print(drums)
        print(guitar)
        
        band_buddy_msg.send_webserver_data(socket_fd, int(genre), int(timbre), int(tempo), float(temperature), drums, guitar, int(bars), float(velocity), band_buddy_msg.STAGE1, band_buddy_msg.WEB_SERVER_STAGE)
        band_buddy_msg.send_webserver_data(socket_fd, int(genre), int(timbre), int(tempo), float(temperature), drums, guitar, int(bars), float(velocity), band_buddy_msg.STAGE2, band_buddy_msg.WEB_SERVER_STAGE)
        band_buddy_msg.send_webserver_data(socket_fd, int(genre), int(timbre), int(tempo), float(temperature), drums, guitar, int(bars), float(velocity), band_buddy_msg.STAGE3, band_buddy_msg.WEB_SERVER_STAGE)
    
        form = GenreForm(request.POST or None, genre=genre,timbre=timbre,tempo=tempo,temperature=temperature,drums=drums,guitar=guitar,bars=bars,velocity=velocity) 
    
    context = { 
        "form":form
    }

    return render(request, "template2.html", context)
