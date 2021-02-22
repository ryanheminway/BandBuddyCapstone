# Magenta model code

This directory contains code targeting the Magenta MusicVAE models. There is a set of forked files which describe the MusicVAE model (https://github.com/magenta/magenta/tree/master/magenta/models/music_vae). With these files, we have attempted a couple different tasks.

1) We have scripts and modifications targetting a translation from a Tensorflow 1.x model to a Tensorflow Lite model.
2) We have modified source files targeting a translation from the Tensorflow 1.x model in to a Tensorflow 2.x model.
3) We have modified source files to allow the Tensorflow 1.x model to run on the Jetson Nano. See NanoMagenta folder

Tasks (1) and (2) are still a work in progress.