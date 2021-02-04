#!/bin/bash
if [ -z "$1" ]
  then
    echo "Please specify a genre!"
    exit 1
fi
GENRE=$1
echo $GENRE
set -x
mkdir "$GENRE"_model_tmp
mkdir "$GENRE"_model
mkdir "$GENRE"_data
cd groove-v1.0.0-midionly
find . -name "*$GENRE*" -exec cp "{}" ../"$GENRE"_data \;
cd ..
convert_dir_to_note_sequences --input_dir="$GENRE"_data --output_file="$GENRE".tfrecord --recursive
music_vae_train --config=groovae_2bar_tap_fixed_velocity --run_dir="$GENRE"_model_tmp --mode=train --examples_path="$GENRE".tfrecord --hparams=batch_size=32, learning_rate=0.0005
cd "$GENRE"_model_tmp
find . -name "*200000*" -exec cp "{}" ../"$GENRE"_model  \;
cd ..
ls
tar -cvf "$GENRE".tar "$GENRE"_model
rm -rf "$GENRE"_model_tmp
rm -rf "$GENRE"_model
rm -rf "$GENRE"_data
rm "$GENRE".tfrecord