# File Structure
There are 3 directories contained in this bundle:
- "datasets" : Contains the "rock.tfrecord" which is a preprocessed dataset to train with
- "magenta" : Contains all the code and should be the root directory for running scripts. Hardcoded relative paths may break otherwise.
- "model_checkpoints" : Contains checkpoints and stored graph definitions of previously trained models.

NOTE: You will see `.dll` files in the magenta directory, and these can simply be ignored. They are required for the `pyfluidsynth` library used in audio processing sections of the code. Irrelevant to the model code.

# Environment
We run a Python environment with Python 3.7 and Tensorflow 2.4.1. Provided in this bundle is an "environment.yml" file which defines the "BB" environment which should have all necessary packages to run our code. Assuming you have Anaconda installed, you can create this environment with the following command:
`conda env create -f environment.yml` and activate with `conda activate BB`

# Note on Magenta code and changes made so far
Most of the code found in this bundle is forked directly from Magenta's github (https://github.com/magenta/magenta/tree/master/magenta/models/music_vae). I have made some changes
and written some utility scripts of my own (`convert_to_tflite.py`, `optimize_for_inference.py`, `inspect_checkpoints.py`, `demo_groovae.py`). I have tried to comment anywhere I made changes to previous Magenta code. 
You can look for the `(NOTE Ryan Heminway)` comment headers. The key change I have attempted so far is using a new definition for the LSTM Cells used by the model. See my comments in `lstm_utils.py`. 
The model still functions properly with this change, but has not fully fixed our TF-Lite conversion errors.

# Training 
To train a new model, we can use the `music_vae_train.py` script provided by magenta. From the `magenta` directory, you can run the following command to start a training session:

`python music_vae_train.py --config=groovae_2bar_tap_fixed_velocity --run_dir=./rundir --mode=train --examples_path=./../datasets/rock.tfrecord --hparams=batch_size=32, learning_rate=0.0005`

For recent experiments, I train only for a short time. This is sufficient since we are more concerned with model graph structure at the moment than we are with model proficiency. 
Training will populate the directory `magenta/rundir/train` with training events, checkpoints, and a model graph. I often call Tensorboard on this directory (`tensorboard --logdir ./rundir/train`) to inspect the model graph. 
Based on inspection of the graph, it seems much of the remaining incompatible TF operators are operators used just for training. 
I have written `optimize_for_inference.py` in an attempt to trim the model graph and remove all training nodes. The hope is this will remove incompatible TF operators while maintaining operators required for inference. 

# Inference
Once the model has been trained, Magenta's code includes `TrainedModel.py` as an interface for interacting with previously trained models.
Since I do not have a lot of experience with Tensorflow, one question I have is about the viability of this type of wrapper class.
It provides multiple methods for "encode" and "decoder" rather than just a single method to perform inference directly. Is this acceptable? Do I need to restructure this interface to produce a model with a single pathway for data?

Either way, the script `demo_groovae.py` is a demonstration of using this TrainedModel to perform inference. There is a single data sample in this bundle, `magenta/basic_plain.wav` which is an unprocessed input for the model. 
The `demo_groovae.py` script does some audio processing to create a processed NoteSequence buffer which is the input for the `TrainedModel`'s `encode` function. There is quite a bit of audio processing code in the `demo_groovae.py` script, but the interaction with the model happens at the bottom of the file. 
The `drumify()` method invokes the `encode` and `decode` functions of the `TrainedModel` to perform inference. 

At the bottom of `demo_groovae.py`, the `GROOVAE_2BAR_TAP_FIXED_VELOCITY` variable defines the path to `.tar` file containing the model checkpoints to load from. It is currently hardcoded to use a `.tar` bundle previously created on a recently trained model (using the latest code). Feel free to update this to point to a new `.tar` file you may produce from your own training runs. 

You can then run the script with `python demo_groovae.py`. No other inputs needed. 
You can assume that the model has built and performed properly if you see the final printout "ALL DONE".
Since we are currently not concerned with model proficiency, it is not necessary to inspect the produced drum files.

By invoking this script, and subsequently building a `TrainedModel`, you will be saving the `TrainedModel`'s graph and session to a `magenta/saved_test_model/saved_model.pb` SavedModel format. I added this to the `TrainedModel` constructor (see line 160 of `TrainedModel.py`). You will see errors on subsequent runs of this script if you have an existing, non-empty, `magenta/saved_test_model` directory. I simply delete the directory when I want to run again. The saving of the model to a SavedModel format was done with the hopes of directing coverting the `saved_model.pb` model to a TF-Lite model using the TF-Lite converter. See the following section for more details.

# TF-Lite Conversion
I have written the `convert_to_tflite.py` script in an attempt to isolate the TF-Lite conversion code. I have tried many different versions of this file, using different model formats (frozen graph, SavedModel, MetaGraphDef), with varying results. None have been able to fully convert the model.

When I run the convert script (`python convert_to_tflite.py`) in its current state, I get the following error:

`tensorflow.lite.python.convert.ConverterError: <unknown>:0: error: loc(callsite("TensorArrayV2Write_2/TensorListSetItem@decoder_while_body_92" at "decoder/while")): failed to legalize operation 'tf.TensorListSetItem' that was explicitly marked illegal
<unknown>:0: note: loc("decoder/while"): called from`

Appears to me that we still have illegal TF operators, but I am unsure how to go about tracking down and removing them. 
Furthermore, I get different results when using different versions of this `convert_to_tflite.py` script.
The current form uses `tensorflow.lite.TFLiteConverter`, but I get different errors when using `tf.compat.v1.lite.TFLiteConverter`. 
When I use this converter instead (the tf 1.x version), I get an error along the lines of:

`tensorflow.lite.python.convert.ConverterError: Graph does not contain node: controls` 

Where `controls` is one of the named inputs I defined in the `TrainedModel` constructor. Removing the names of the inputs does not fix this issue. It is unclear to me which approach is correct, since both the 1.x and 2.x converters are throwing errors. 

It is also worth nothing that the current version of the script does not attempt the freeze the graph. It simply uses the `saved_model.pb` produced by the `TrainedModel` constructor.