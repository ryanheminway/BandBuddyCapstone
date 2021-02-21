"""
Author: Ryan Heminway
Attempting to trim an existing graph to remove training nodes and operators. Some of the training ops seem to be
incompatible with TF-Lite and should be removed once the model is trained since we only need the model for inference
at that point.
"""

import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

# (NOTE Ryan Heminway) Unclear to me if this is the best approach. It appears to work, creating a OptimizedGraph.pb file,
#   but the TF-Lite converter fails to recognize that file as a valid Saved Model
with tf.Session() as sess:
    model_filename ='saved_model.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
        print(type(sm.meta_graphs[0].graph_def))
        # Removes nodes only used for training. Hopefully remove some incompatible Ops
        output_graph = optimize_for_inference(sm.meta_graphs[0].graph_def, ["temperature", "z_input", "inputs", "controls", "input_length", "max_length"], ["outputs"], tf.float32.as_datatype_enum)
        print(type(output_graph))
        # Save the optimized graph'test.pb'
        new_f = tf.gfile.FastGFile('OptimizedGraph.pb', "w")
        new_f.write(output_graph.SerializeToString())
        # (NOTE) This actually works, but the TFLite converter fails when given this .pb file as a saved model