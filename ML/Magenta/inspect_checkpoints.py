"""
Author: Ryan Heminway
Quality of life script to inspect the checkpoint file of a model. Helpful for determining the tensor / op names in
a graph as well as comparing checkpoints.

Created to investigate differences between old magenta models (came as pre-trained) and new ones that we have trained
with code modifications.

Filepaths are likely outdated and need updating after each training session
"""

from tensorflow.python.tools import inspect_checkpoint

# Checkpoint for the model that came pretrained with magenta
old_magenta_ckpt = "../model_checkpoints/groovae_2bar_tap_fixed_velocity/model.ckpt-3668"
# Checkpoint for the model that I trained with (supposedly) new architecture
new_magenta_ckpt = "../model_checkpoints/groovae_rock/model.ckpt-12804"
# Checkpoint for the model that I trained with old architecture
new_old_magenta_ckpt = "../model_checkpoints/groovae_rock/old_code/model.ckpt-1556"

print("OLD MODEL ::::::: \n\n")
inspect_checkpoint.print_tensors_in_checkpoint_file(old_magenta_ckpt, None, False, True)

print("NEW MODEL ::::::: ")
inspect_checkpoint.print_tensors_in_checkpoint_file(new_magenta_ckpt, None, False, True)

#print("NEW OLD MODEL ::::::: ")
#inspect_checkpoint.print_tensors_in_checkpoint_file(new_old_magenta_ckpt, None, False, True)

