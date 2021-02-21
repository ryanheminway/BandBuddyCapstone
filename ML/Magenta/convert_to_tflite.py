"""
Author: Ryan Heminway
This file is an attempt to take a trained model (via SavedModel, or Frozen Graph) and
convert it to a TF-Lite model using the TF-Lite converter
Converter documentation: https://www.tensorflow.org/lite/convert/
"""
import os
import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
from tensorflow.python.tools import freeze_graph

MODEL_NAME = 'groovae_rock'
input_graph_path = './../model_checkpoints/groovae_rock/graph.pbtxt'
checkpoint_path = './../model_checkpoints/groovae_rock/model.ckpt-12804'
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'

need_freezing = False
convert_tflite = True

if need_freezing:
    freeze_graph.freeze_graph(input_graph_path, input_saver="",
                          input_binary=True, input_checkpoint=checkpoint_path,
                          output_node_names="encoder/sigma/BiasAdd", restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0",
                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")

# Attempting with saved model
saved_model_path = "./saved_test_model"

# (NOTE Ryan Heminway) Very unclear to me which is the appropriate way to convert to TF-Lite.
#   Do we use 1.x converter since most of the tensorflow code is still in 1.x style?
#   Do we use 2.x converter since the LSTM cells now use a 2.x (Keras LSTMCell) definition?
#   Is it preferrable to convert from frozen graph? or Saved Model?
#   Do we need to target a specific set of supported ops?
if convert_tflite:
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model_path)
    #converter = tensorflow.lite.TFLiteConverter.from_saved_model(saved_model_path)
    #converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(output_frozen_graph_name, ["Placeholder_2"], ["encoder/sigma/BiasAdd"])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()