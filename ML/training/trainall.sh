#!/bin/bash
set -x
mkdir model_tmp
mkdir model
mkdir data
cd groove-v1.0.0-midionly
find . -name "*.mid*" -exec cp "{}" ../data \;
cd ..
convert_dir_to_note_sequences --input_dir=data --output_file=record.tfrecord --recursive
music_vae_train --config=groovae_2bar_tap_fixed_velocity --run_dir=model_tmp --mode=train --examples_path=record.tfrecord --hparams=batch_size=32, learning_rate=0.0005
cd model_tmp
find . -name "*200000*" -exec cp "{}" ../model  \;
cd ..
ls
tar -cvf all.tar model
rm -rf model_tmp
rm -rf model
rm -rf data
rm record.tfrecord