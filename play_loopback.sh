#!/bin/bash

# If the loopback file exists, play it
if [[ -f "loopback.wav" ]]; then 
    aplay -D plughw:CARD=pisound -f S16  loopback.wav
else 
    echo "Omitted .wav file from repo; scp any stereo 48000KHz wav file to the calling directory and name it loopback.wav"
fi