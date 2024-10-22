# Data

## Groove MIDI Dataset

The base dataset that we can use. 

https://magenta.tensorflow.org/datasets/groove

## Expanded Groove MIDI Dataset 

In march, they released a data set 40x the size of ^^. This is great, gives us a ton of data.

https://magenta.tensorflow.org/datasets/e-gmd

# General Resources

Magenta overview. Links to guides on how to set up the development environment for magenta projects, etc. Will be useful when working with ML component.

https://github.com/magenta/magenta/blob/master/README.md

GrooVAE demonstration in Jupyter notebook. Note that we do not need the full suite of functionality that "GrooVAE" gives birth to. We may only use Tap2Drum which goes from a rhythm to a drum track.

https://colab.research.google.com/github/tensorflow/magenta-demos/blob/master/colab-notebooks/GrooVAE.ipynb

List of references for general music processing in software. Havn't gone through it all.

http://ismir.net/resources/software-tools/

Similar, but more specific. Has some guides on general music processing. The processes they explain here aren't necessarily the way we are going to do it, but listing it here anyway. (Also a great resource for an introduction to using Jupyter Notebooks) 

https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html

# MIDI to WAV

Have not tried this at all, but looks promising. Python package for MIDI synthesis to .wav

https://mdoege.github.io/PySynth/

Python implementation for FluidSynth, which is a software MIDI Synthesizer. Very promising because FluidSynth is the backend for VLC Media player, which is able to synthesize and play midi files on windows. 

Also available as a pip package.

https://pypi.org/project/pyFluidSynth/

# WAV to MIDI 

Possible solution. Again... havn't really tried it yet. Tried other ones that didn't work.

https://pypi.org/project/audio-to-midi/

Found how Magenta does their "audio to midi" translation. They use this Librosa library to extract timing information about notes. They ignore the pitch of each note because they really don't need it for their drum model. See the MagentaDemo notebook for more information.

https://librosa.org/doc/latest/index.html

List of WAV to MIDI resources

https://gist.github.com/natowi/d26c7e97443ec97e8032fb7e7596f0b0

# MIDI resources / packages 

https://pypi.org/project/MIDIFile/

https://craffel.github.io/pretty-midi/









