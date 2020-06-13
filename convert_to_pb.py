import numpy as np
import tensorflow as tf
if tf.__version__ == '2.1.0':
    import tensorflow.compat.v1.keras.backend as K
else:
    import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import sys

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def convert_to_pb(model_in_memory, output_file=None):
    model = model_in_memory
    frozen_graph = freeze_session(K.get_session(), 
        output_names=[out.op.name for out in model.outputs])
    if output_file is None:
        output_file = model.name + '.pb'
        if tf.__version__ not in output_file:
            output_file = model.name + tf.__version__ + '.pb'
    tf.train.write_graph(frozen_graph, './', output_file, as_text=False)