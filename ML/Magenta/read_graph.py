import tensorflow.compat.v1 as tf
import sys
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

with tf.Session() as sess:
    model_filename ='saved_model.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
        print(type(sm.meta_graphs[0].graph_def))
        output_graph = optimize_for_inference(sm.meta_graphs[0].graph_def, ["temperature", "z_input", "inputs", "controls", "input_length", "max_length"], ["outputs"], tf.float32.as_datatype_enum)
        print(type(output_graph))