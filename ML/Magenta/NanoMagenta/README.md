# Magenta files for the Jetson Nano

All files required to run the Magenta model on the Jetson Nano. The magenta files have been modified (still a WIP) to remove filler code and remove as many dependency libraries as possible. To my knowledge, the only 3 major libraries required for this to run are Tensorflow 2.3.1, Tensorflow-Probability 0.11.1, and Note-seq. Most importantly, I have removed all mention of the Magenta library, as it is not installable on the Jetson Nano.