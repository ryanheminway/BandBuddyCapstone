"""
Author: Ryan Heminway

Reimplementation of magenta/base_model.py to use Tensorflow 2.x rather than 1.x functionality.
"""
import collections
import os

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import tensorflow_addons as tfa

import configs
import configs as cfg
import data

#import lstm_utils

"""========================= START HELPERS =================================="""

def flatten_maybe_padded_sequences(maybe_padded_sequences, lengths=None):
  """Flattens the batch of sequences, removing padding (if applicable).

  Args:
    maybe_padded_sequences: A tensor of possibly padded sequences to flatten,
        sized `[N, M, ...]` where M = max(lengths).
    lengths: Optional length of each sequence, sized `[N]`. If None, assumes no
        padding.

  Returns:
     flatten_maybe_padded_sequences: The flattened sequence tensor, sized
         `[sum(lengths), ...]`.
  """
  def flatten_unpadded_sequences():
    # The sequences are equal length, so we should just flatten over the first
    # two dimensions.
    return tf.reshape(maybe_padded_sequences,
                      [-1] + maybe_padded_sequences.shape.as_list()[2:])
  if lengths is None:
    return flatten_unpadded_sequences()
  def flatten_padded_sequences():
    indices = tf.where(tf.sequence_mask(lengths))
    return tf.gather_nd(maybe_padded_sequences, indices)
  return tf.cond(
      tf.equal(tf.reduce_min(lengths), tf.shape(maybe_padded_sequences)[1]),
      flatten_unpadded_sequences,
      flatten_padded_sequences)


class LstmDecodeResults(collections.namedtuple('LstmDecodeResults',
                                               ('rnn_output', 'samples',
                                                'final_state', 'final_sequence_lengths'))):
  pass


class Seq2SeqLstmDecoderOutput(
    collections.namedtuple('BasicDecoderOutput',
                           ('rnn_input', 'rnn_output', 'sample_id'))):
  pass


class Seq2SeqLstmDecoder(tfa.seq2seq.BasicDecoder):
  """Overrides BaseDecoder to include rnn inputs in the output.
  (NOTE Ryan H):
  """

  def __init__(self, cell, sampler, initial_state, input_shape,
               output_layer=None):
    self._input_shape = input_shape
    self._initial_state = initial_state
    super(Seq2SeqLstmDecoder, self).__init__(
        cell, sampler, output_layer)

  @property
  def output_size(self):
    return Seq2SeqLstmDecoderOutput(
        rnn_input=self._input_shape,
        rnn_output=self._rnn_output_size(),
        sample_id=self.sampler.sample_ids_shape)

  @property
  def output_dtype(self):
    dtype = tf.nest.flatten(self._initial_state)[0].dtype
    return Seq2SeqLstmDecoderOutput(
        dtype,
        tf.nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        self.sampler.sample_ids_dtype)

  def step(self, time, inputs, state, training, name="seq2seq_decoder"):
    results = super(Seq2SeqLstmDecoder, self).step(time, inputs, state, training, name)
    outputs = Seq2SeqLstmDecoderOutput(
        rnn_input=inputs,
        rnn_output=results[0].rnn_output,
        sample_id=results[0].sample_id)
    return (outputs,) + results[1:]


# Computes an initial RNN Cell state from embedding 'z'
def initial_state_from_embedding(cell, z, name="z_initial_state"):
    flat_state_sizes = tf.nest.flatten(cell.state_size)
    z_initial_layer = tfk.layers.Dense(sum(flat_state_sizes),
                                       activation=tf.tanh,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                       name=name)
    initial_state = tf.nest.pack_sequence_as(cell.get_initial_state(batch_size=z.shape[0], dtype=tf.float32),
                                             tf.split(z_initial_layer(z),
                                                      flat_state_sizes,
                                                      axis=1))
    return initial_state


"""========================= END HELPERS =================================="""

"""========================= START DECODER =================================="""

# TF 2.x implementation of Magenta's GrooveLstmDecoder (lstm_models.py/GrooveLstmDecoder)
class GrooVAEDecoder(tfk.layers.Layer):
    def __init__(self, hparams, temperature, output_depth, is_training, name="decoder", **kwargs):
        super(GrooVAEDecoder, self).__init__(name=name, **kwargs)
        self.hparams = hparams
        self.max_seq_len = hparams.max_seq_len
        self.output_depth = output_depth
        self.temperature = temperature
        self.sampler = None #  Diff sampler used in training vs inference
        self.inp_shape = None # Diff input shape used in training vs inference (I think?)
        self.dropout_prob = (1 - self.hparams.dropout_keep_prob) if is_training else 0
        dec_rnn_size = hparams.dec_rnn_size
        dec_cells = []
        # dec_rnn_size = [256, 256]
        for i in range(len(dec_rnn_size)):
            lstm_cell = tfk.layers.LSTMCell(dec_rnn_size[i], dropout=self.dropout_prob)
            dec_cells.append(lstm_cell)
        self.dec_cell = tfk.layers.StackedRNNCells(dec_cells)
        self.output_layer = tfk.layers.Dense(self.output_depth, name="output_projection")

    # Decodes from sample from latent vector z and decoder input x_input\
    # CALL_INPUTS = [z, x_input]
    def call(self, call_inputs):
        z = call_inputs[0]
        x_input = call_inputs[1]
        sampler = self.sampler
        input_shape = self.inp_shape

        init_state = initial_state_from_embedding(self.dec_cell, z)
        print("INIT STATE: ", init_state)
        print("Z: ", z)
        print("INPUT SHAPE: ", input_shape)
        print("X INPUT: ", x_input)

        """ I get problems with Magenta's custom Seq2Seq LSTM decoder. 
        Issues with output type and shape
        
        decoder = Seq2SeqLstmDecoder(
            self.dec_cell,
            sampler,
            input_shape=input_shape,
            initial_state=init_state,
            output_layer=self.output_layer)"""

        decoder = tfa.seq2seq.BasicDecoder(parallel_iterations=1,
            input_shape=input_shape,
            cell=self.dec_cell,
            sampler=sampler,
            output_layer=self.output_layer)
        final_output, final_state, final_lengths = tfa.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=None,
            swap_memory=True,
            scope='decoder',
            decoder_init_input=x_input,
            decoder_init_kwargs={'initial_state': init_state })
        # (TODO) enable flag once training works without it
        # NOTE !!!! tfa.seq2seq.dynamic_decode has enable_tflite_convertible flag !!!!

        results = LstmDecodeResults(
            rnn_output=final_output.rnn_output,
            samples=final_output.sample_id,
            final_state=final_state,
            final_sequence_lengths=final_lengths)

        return results

    # Generate sample from decoder based on latent vector z
    def sample(self, z):
        start_inputs = tf.zeros([1, self.output_depth], dtype=tf.dtypes.float32)
        # In the conditional case, also concatenate the Z.
        start_inputs = tf.concat([start_inputs, z], axis=-1)
        initialize_fn = lambda: (tf.zeros([1], tf.bool), start_inputs)

        sample_fn = lambda time, outputs, state: self._sample(outputs, self.temperature)
        end_fn = (lambda x: False)

        def next_inputs_fn(time, outputs, state, sample_ids):
            del outputs
            finished = end_fn(sample_ids)
            next_inputs = tf.concat([sample_ids, z], axis=-1)
            return (finished, next_inputs, state)

        # Custom sampler, no teacher forcing when we are sampling. Not used in training
        sampler = tfa.seq2seq.CustomSampler(
            initialize_fn=initialize_fn, sample_fn=sample_fn,
            next_inputs_fn=next_inputs_fn, sample_ids_shape=[self.output_depth],
            sample_ids_dtype=tf.dtypes.float32)

        self.sampler = sampler
        self.input_shape = start_inputs.shape[1:]
        # (TODO) Might be calling with wrong inputs
        decode_results = self.call([z, start_inputs])

        return decode_results.samples

    def _activate_outputs(self, flat_rnn_output):
        output_hits, output_velocities, output_offsets = tf.split(
            flat_rnn_output, 3, axis=1)

        output_hits = tf.sigmoid(output_hits)
        output_velocities = tf.sigmoid(output_velocities)
        output_offsets = tf.tanh(output_offsets)

        return output_hits, output_velocities, output_offsets

    def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
        # flat_x_target is by default shape (1,27), [on/offs... vels...offsets...]
        # split into 3 equal length vectors
        target_hits, target_velocities, target_offsets = tf.split(
            flat_x_target, 3, axis=1)

        output_hits, output_velocities, output_offsets = self._activate_outputs(
            flat_rnn_output)

        hits_loss = tf.reduce_sum(tf.compat.v1.losses.log_loss(
            labels=target_hits, predictions=output_hits,
            reduction=tf.losses.Reduction.NONE), axis=1)

        velocities_loss = tf.reduce_sum(tf.compat.v1.losses.mean_squared_error(
            target_velocities, output_velocities,
            reduction=tf.losses.Reduction.NONE), axis=1)

        offsets_loss = tf.reduce_sum(tf.compat.v1.losses.mean_squared_error(
            target_offsets, output_offsets,
            reduction=tf.losses.Reduction.NONE), axis=1)

        loss = hits_loss + velocities_loss + offsets_loss

        return loss

    def _sample(self, rnn_output, temperature=1.0):
        output_hits, output_velocities, output_offsets = tf.split(
            rnn_output, 3, axis=1)

        output_velocities = tf.sigmoid(output_velocities)
        output_offsets = tf.tanh(output_offsets)

        hits_sampler = tfp.distributions.Bernoulli(
            logits=output_hits / temperature, dtype=tf.dtypes.float32)

        output_hits = hits_sampler.sample()
        return tf.concat([output_hits, output_velocities, output_offsets], axis=1)

    # Compute loss term for how well decoder was able to properly construct target output sequence
    def reconstruction_loss(self, x_input, x_target, x_length, z):
        batch_size = int(x_input.shape[0])
        print("BATCH_SIZE IN REC LOSS: ", batch_size)

        """
        TF 1.x impl passes x_input to Sampler (called helper in 1.x). 2.x Sampler 
        doesn't seem to use it"""
        
        repeated_z = tf.tile(
            tf.expand_dims(z, axis=1), [1, tf.shape(x_input)[1], 1])
        x_input = tf.concat([x_input, repeated_z], axis=2)


        # (NOTE) removed Control input from here. Appeared to be "None" in all contexts. Do we need it?

        # Use teacher forcing
        self.sampler = tfa.seq2seq.TrainingSampler()
        # Correct input shape? 1.x model it is (283,)
        self.inp_shape = (283,) # self.sampler.inputs.shape[2:] # doesn't exist

        decode_results = self.call([z, x_input])
        flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
        flat_rnn_output = flatten_maybe_padded_sequences(
            decode_results.rnn_output, x_length)
        r_loss = self._flat_reconstruction_loss(
            flat_x_target, flat_rnn_output)

        # Sum loss over sequences.
        cum_x_len = tf.concat([(0,), tf.cumsum(x_length)], axis=0)
        r_losses = []
        for i in range(batch_size):
            b, e = cum_x_len[i], cum_x_len[i + 1]
            r_losses.append(tf.reduce_sum(r_loss[b:e]))
        r_loss = tf.stack(r_losses)

        return r_loss


"""========================= END DECODER =================================="""

"""========================= START GROOVAE MODEL =================================="""
"""
TF 2.x implementation of Magenta GrooVAE model. Uses Bidirectional LSTM Encoder and 
LSTM Decoder. Decoder is a bit more complicated so it is given its own class implementation.
Decoder uses teacher forcing during training. 
"""
class GrooVAE(tfk.Model):
    def __init__(self, hparams, output_depth, is_training, name="vae", **kwargs):
        super(GrooVAE, self).__init__(name=name, **kwargs)
        self.is_training = is_training
        self.hparams = hparams
        self.output_depth = output_depth
        # using 1 - hparams.dropout because Keras dropout is inverse of old rnn.DropoutWrapper
        self.dropout_prob = (1 - self.hparams.dropout_keep_prob) if self.is_training else 0
        # GrooVAE (2bar_tap_fixed_velocity) Config dictates NO KL annealing, so weight is constant
        self.kl_weight = self.hparams.max_beta
        self.encoder = self.encoder_z()
        # (TODO) temperature should be configurable. For now doesn't matter
        self.decoder = GrooVAEDecoder(hparams=self.hparams, temperature=1.0,
                                      is_training=self.is_training,  output_depth=self.output_depth)

    # Sequential API (Bidirectional LSTM) Encoder
    def encoder_z(self):
        layers = []
        # enc_rnn_size = [512]
        lstm = tfk.layers.LSTM(self.hparams.enc_rnn_size[0], dropout=self.dropout_prob)
        # (NOTE Ryan Heminway) Does this Bidirectional return outputs in same ordering
        #   that previous impl did?
        layers.append(tfk.layers.Bidirectional(lstm))
        return tfk.Sequential(layers, name="encoder_z")

    # Encode to Z distribution (MultivariateNormalDiag distribution)
    def encode(self, input_sequence):
        enc_output = self.encoder(input_sequence)
        print("enc_output: ", enc_output)
        mu_layer = tfk.layers.Dense(
            self.hparams.z_size,
            kernel_initializer=tfk.initializers.RandomNormal(stddev=0.001),
            name="encoder/mu_layer")
        mu = mu_layer(enc_output)
        print("mu: ", mu)
        sigma_layer = tfk.layers.Dense(
            self.hparams.z_size,
            activation=tfk.activations.softplus,
            kernel_initializer=tfk.initializers.RandomNormal(stddev=0.001),
            name="encoder/sigma_layer")
        sigma = sigma_layer(enc_output)
        print("sigma: ", sigma)
        print("same? :", tf.math.equal(mu, sigma))
        z = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        print("Z DISTRIBUTION: ", z)
        z_sample = z.sample()
        print("Z SAMPLE: ", z)
        return (z, z_sample)

    # Encoder sequence to Z distribution and then decode to output sequence
    def call(self, x_input):
        print("input seq: ", x_input)
        # Real Z distribution produced by encoder
        (q_z, q_z_sample) = self.encode(x_input)
        # Prior distribution
        p_z = tfp.distributions.MultivariateNormalDiag(loc=[0.] * self.hparams.z_size,
                                                       scale_diag=[1.] * self.hparams.z_size)
        # KL Divergence (measure relative difference between distributions)
        kl_div = tfp.distributions.kl_divergence(q_z, p_z)
        # (NOTE Ryan Heminway) Not quite sure what this free_nats business is about
        #       Copying from 1.x version
        free_nats = self.hparams.free_bits * tf.math.log(2.0)
        kl_cost = tf.maximum(kl_div - free_nats, 0)
        kl_cost = self.kl_weight * tf.reduce_mean(kl_cost)
        self.add_loss(kl_cost) # KL Divergence as loss term
        # (TODO) Correct call arguments?
        output_seq = self.decoder([q_z_sample, x_input])
        return output_seq


"""========================= END GROOVAE MODEL =================================="""

"""========================= START TRAINING FUNCTIONS =================================="""

"""
(NOTE Ryan H): 
Currently, I don't know where to call the GrooVAE "call" method. The reconstruction loss for 
the decoder does not require the use of the call method on the GrooVAE model. This is because
the reconstruction loss requires the Z distribution from the model, which is not output
if you call the whole model
"""


# Reconstruction loss for decoder training
def partial_vae_loss(input_seq, output_seq, seq_length, groove_model):
    (_, z) = groove_model.encode(input_seq)
    reconstruct_loss = groove_model.decoder.reconstruction_loss(input_seq, output_seq, seq_length, z)
    r_loss = tf.reduce_mean(reconstruct_loss)
    return r_loss


# Single step of training. Compute reconstruction loss, add to KL loss, and compute gradients
@tf.function # (TODO) I get different errors when marking this as a tf.function
def train_step(input_seq, output_seq, seq_length, groove_model, train_optimizer, train_loss_metric):
    with tf.GradientTape() as tape:
        # _ = groove_model(input_seq) # ?? I don't know how I am supposed to call model
        r_loss = partial_vae_loss(input_seq, output_seq, seq_length, groove_model)
        kl_loss = tf.math.reduce_sum(groove_model.losses)  # vae.losses is a list
        total_vae_loss = r_loss + kl_loss
    gradients = tape.gradient(total_vae_loss, groove_model.trainable_variables)
    train_optimizer.apply_gradients(zip(gradients, groove_model.trainable_variables))
    train_loss_metric(total_vae_loss)


"""========================= END TRAINING FUNCTIONS =================================="""

"""========================= START TRAINING / MODEL PARAMS =================================="""


epochs = 10  # (TODO) Arbitrarily low number for now, until training is working
groovae_cfg = cfg.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
lr = 0.0005 #(TODO) Currently no learning-rate decay. Should impl once training working
data_converter = groovae_cfg.data_converter
run_dir = "./../datasets/rundir/"
run_dir = os.path.expanduser(run_dir)
train_dir = os.path.join(run_dir, 'train')
data_record = "./../datasets/rock.tfrecord"
tf_file_reader = tf.data.TFRecordDataset
file_reader = tf.compat.v1.python_io.tf_record_iterator
config_update_map = {'train_examples_path': os.path.expanduser(data_record)}
groovae_cfg = configs.update_config(groovae_cfg, config_update_map)

model = GrooVAE(groovae_cfg.hparams, data_converter.output_depth, True)
#model.build(input_shape=(1, 32, 27))
#model.summary()
optimizer = tfk.optimizers.Adam(lr)
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


"""========================= END DATA FUNCTIONS =================================="""

#print("INPUT DATA ? : ", _get_input_tensors(dataset_fn(), groovae_cfg))

"""========================= START TRAIN LOOP =================================="""

for epoch in range(epochs):
    # Get next batch of tensor data to train on
    batch_data = _get_input_tensors(dataset_fn(), groovae_cfg)
    print("INPUT SEQ: ", tf.shape(batch_data["input_sequence"]))

    # Some data processing as in base_model.py/MusicVAE/_compute_model_loss
    input_sequence = tf.cast(batch_data["input_sequence"], dtype=tf.float32)
    output_sequence = tf.cast(batch_data["output_sequence"], dtype=tf.float32)
    max_sequence_length = tf.minimum(tf.shape(output_sequence)[1], groovae_cfg.hparams.max_seq_len)
    input_sequence = input_sequence[:,:max_sequence_length]
    output_sequence = output_sequence[:, :max_sequence_length]
    sequence_length = tf.minimum(batch_data["sequence_length"], max_sequence_length)

    # Train step
    train_step(input_sequence, output_sequence, sequence_length, model, optimizer, loss_metric)
    elbo = -loss_metric.result()
    print('Epoch: {}, Train set ELBO: {}'.format(
           epoch, elbo))

print("DONE TRAINING LOOP")
"""========================= END TRAIN LOOP =================================="""

# generate .tflite file
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_save = converter.convert()

