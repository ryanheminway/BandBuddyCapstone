import os

import tensorflow as tf
import tensorflow.keras as tfk

import configs as cfg
import data
from base_model_v2 import GrooVAE, reconstruction_loss

# Disables cuda
# (TODO) Try with cuda again once model.save is working. Currently CUDA doesn't like our training procedure
os.environ["CUDA_VISIBLE_DEVICEs"] = "-1"


# Custom loss term that returns a function which imitates the form loss(input, expected_output) which
# is required by Keras model.compile and subsequently model.build
@tf.function
def loss_term(input_seq, output_seq, seq_length, groove_model):
    (z, z_sample) = groove_model.encoder(input_seq)
    reconstruct_loss = reconstruction_loss(groove_model.decoder, input_seq, output_seq, seq_length, z_sample)
    r_loss = tf.reduce_mean(reconstruct_loss)
    #kl_loss = tf.math.reduce_sum(groove_model.losses)  # vae.losses is a list
    # Separate function (not model.loss term) to avoid accessing graph tensor
    kl_loss = groove_model.kl_loss(z)
    total_vae_loss = r_loss + kl_loss
    return total_vae_loss
    #return lambda input, output: total_vae_loss


# Single step of training. Compute reconstruction loss, add to KL loss, and compute gradients
# (TODO) I get errors when marking this as a tf.function
def train_step(input_seq, output_seq, seq_length, groove_model, train_optimizer, train_loss_metric):
    with tf.GradientTape() as tape:
        loss_value = loss_term(input_seq, output_seq, seq_length, groove_model)
        # (TODO) hacky and dumb and I hate it
        #loss_value = loss_fn(input_seq, output_seq)
    gradients = tape.gradient(loss_value, groove_model.trainable_variables)
    train_optimizer.apply_gradients(zip(gradients, groove_model.trainable_variables))
    train_loss_metric(loss_value)


"""========================= END TRAINING FUNCTIONS =================================="""

"""========================= START TRAINING / MODEL PARAMS =================================="""

epochs = 3  # (TODO) Arbitrarily low number for now, until training is working
groovae_cfg = cfg.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
lr = groovae_cfg.hparams.learning_rate
data_converter = groovae_cfg.data_converter
run_dir = "./../datasets/rundir/"
run_dir = os.path.expanduser(run_dir)
train_dir = os.path.join(run_dir, 'train')
data_record = "./../datasets/rock.tfrecord"
tf_file_reader = tf.data.TFRecordDataset
file_reader = tf.compat.v1.python_io.tf_record_iterator
config_update_map = {'train_examples_path': os.path.expanduser(data_record)}
groovae_cfg = cfg.update_config(groovae_cfg, config_update_map)

model = GrooVAE(groovae_cfg.hparams, data_converter.output_depth, True)

optimizer = tfk.optimizers.Adam(lr)
#model.compile(optimizer, loss=loss_term())
#model.build(input_shape=(1,32,27))
loss_metric = tfk.metrics.Mean() # Sum?

"""========================= END TRAINING / MODEL PARAMS =================================="""

"""========================= START DATA FUNCTIONS =================================="""

def dataset_fn():
    return data.get_dataset(
        groovae_cfg,
        tf_file_reader=tf_file_reader,
        is_training=True,
        cache_dataset=True)

def _get_input_tensors(dataset, config):
  """Get input tensors from dataset."""
  batch_size = config.hparams.batch_size
  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
  (input_sequence, output_sequence, control_sequence,
   sequence_length) = iterator.get_next()
  input_sequence.set_shape(
      [batch_size, None, config.data_converter.input_depth])
  output_sequence.set_shape(
      [batch_size, None, config.data_converter.output_depth])
  if not config.data_converter.control_depth:
    control_sequence = None
  else:
    control_sequence.set_shape(
        [batch_size, None, config.data_converter.control_depth])
  sequence_length.set_shape([batch_size] + sequence_length.shape[1:].as_list())

  return {
      'input_sequence': input_sequence,
      'output_sequence': output_sequence,
      'control_sequence': control_sequence,
      'sequence_length': sequence_length
  }


# "Fix" set of input tensors by reshaping according to depth and batch
def resize_input_tensors(config, input_seq, output_seq, control_seq, seq_len):
    batch_size = config.hparams.batch_size
    input_seq.set_shape(
        [batch_size, None, config.data_converter.input_depth])
    output_seq.set_shape(
        [batch_size, None, config.data_converter.output_depth])
    if not config.data_converter.control_depth:
        control_seq = None
    else:
        control_seq.set_shape(
            [batch_size, None, config.data_converter.control_depth])
    seq_len.set_shape([batch_size] + seq_len.shape[1:].as_list())

    return {
        'input_sequence': input_seq,
        'output_sequence': output_seq,
        'control_sequence': control_seq,
        'sequence_length': seq_len
    }

# Get the learning rate at a given global step
def get_lr(step, cfg):
    lr = ((cfg.hparams.learning_rate - cfg.hparams.min_learning_rate) *
          tf.pow(cfg.hparams.decay_rate, tf.cast(step, dtype=tf.float32)) + cfg.hparams.min_learning_rate)
    return lr


"""========================= END DATA FUNCTIONS =================================="""


"""========================= START TRAIN LOOP =================================="""

step = 0
for epoch in range(epochs):
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset_fn())
    # Every epoch, loop over all batches in training
    for batch in iterator:
        batch_data = resize_input_tensors(groovae_cfg, batch[0], batch[1], batch[2], batch[3])
        #print("INPUT SEQ: ", tf.shape(batch_data["input_sequence"]))

        # Some data processing as in base_model.py/MusicVAE/_compute_model_loss
        input_sequence = tf.cast(batch_data["input_sequence"], dtype=tf.float32)
        output_sequence = tf.cast(batch_data["output_sequence"], dtype=tf.float32)
        max_sequence_length = tf.minimum(tf.shape(output_sequence)[1], groovae_cfg.hparams.max_seq_len)
        input_sequence = input_sequence[:, :max_sequence_length]
        output_sequence = output_sequence[:, :max_sequence_length]
        sequence_length = tf.minimum(batch_data["sequence_length"], max_sequence_length)
        if (sequence_length != max_sequence_length):
            print("Sequence length !! : ", sequence_length)

        optimizer.lr.assign(get_lr(step, groovae_cfg))

        # (TODO) This seems really stupid...
        #if step == 0:
        #    model.compile(optimizer, loss=loss_term(input_seq=input_sequence, output_seq=output_sequence, seq_length=sequence_length, groove_model=model))

        # Train step
        train_step(input_sequence, output_sequence, sequence_length, model, optimizer, loss_metric)
        elbo = -loss_metric.result()
        print('Epoch: {}, Train set ELBO: {}'.format(
              epoch, elbo))
        step += 1
        # model.compile first to allow .build to work
        # model.build to set input shapes for model.save
        # (TODO) we call model.build but .save still fails saying we didn't set input shapes?
        #model.build(input_shape=(1, 32, 27))
        #model.save("./model_test_saved/")
        if epoch % 10 == 0:
            # (TODO) Sometimes this fails due to "folder: access denied"???
            model.save_weights("./model_test_checkpoint/groovae_ckpt")
        if (elbo > -50):
            model.save_weights("./model_test_checkpoint/groovae_ckpt")
            break
print("DONE TRAINING LOOP")


# (TODO) this stuff fails cuz the model isn't really set up for keras it seems

#print("inp shape: ", model.input_shape)
#model.compute_output_shape(input_shape=(1,32,27))
#model.summary()
#model.save("./saved_model")
#loaded_model = tfk.models.load_model("./saved_model")
#noteseq_result = groovae_cfg.data_converter.from_tensors(tensor_result)[0]
#print("noteseq result: ", noteseq_result)
#wav_results = midi_to_wav(noteseq_result, sample_rate=22050)
#sf.write("model_out_drums.wav", wav_results, 22050, subtype='PCM_24')

"""========================= END TRAIN LOOP =================================="""

# generate .tflite file
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_save = converter.convert()
