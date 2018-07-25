import io
import os
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from util.text import text_to_sequence
from util import audio, plot
import textwrap
import sounddevice as sd
import pyaudio
import wave
import datetime


class Synthesizer:
  def __init__(self, teacher_forcing_generating=False):
    self.teacher_forcing_generating = teacher_forcing_generating
  def load(self, checkpoint_path, reference_mel=None, gta=False, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths') 

    if reference_mel is not None:
      reference_mel = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'reference_mel')
    # Only used in teacher-forcing generating mode
    if gta:
      mel_targets = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'mel_targets')
    else:
      mel_targets = None

    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths, mel_targets=mel_targets, reference_mel=reference_mel, gta=gta)
      #self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])
      self.alignment = self.model.alignments[0]
      self.mel_outputs = self.model.mel_outputs
      self.gta= gta
    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text, index, out_dir, log_dir, mel_filename, mel_targets=None, reference_mel=None):

    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
    }

    if mel_targets is not None:
      mel_targets = np.expand_dims(mel_targets, 0)
      feed_dict.update({self.model.mel_targets: np.asarray(mel_targets, dtype=np.float32)})
    if reference_mel is not None:
      reference_mel = np.expand_dims(reference_mel, 0)
      feed_dict.update({self.model.reference_mel: np.asarray(reference_mel, dtype=np.float32)})

    if self.gta is True:
      feed_dict[self.model.mel_targets] = np.load(mel_filename).reshape(1, -1, 80)

    if self.gta or not hparams.predict_linear:
      mels, alignment = self.session.run([self.mel_outputs, self.alignment], feed_dict=feed_dict)


    else:
      linear, mels, alignment = self.session.run([self.linear_outputs, self.mel_outputs, self.alignment], feed_dict=feed_dict)
      linear = linear.reshape(-1, hparams.num_freq)

    mels = mels.reshape(-1, hparams.num_mels) #Thanks to @imdatsolak for pointing this out

    if index is None:

      #Generate wav and read it
      wav = audio.inv_mel_spectrogram(mels.T)
      audio.save_wav(wav, 'temp.wav') #Find a better way

      chunk = 512
      f = wave.open('temp.wav', 'rb')
      p = pyaudio.PyAudio()
      stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
        channels=f.getnchannels(),
        rate=f.getframerate(),
        output=True)
      data = f.readframes(chunk)
      while data:
        stream.write(data)
        data=f.readframes(chunk)

      stream.stop_stream()
      stream.close()

      p.terminate()
      return


    # Write the spectrogram to disk
    # Note: outputs mel-spectrogram files and target ones have same names, just different folders
    mel_filename = os.path.join(out_dir, 'speech-mel-{:05d}.npy'.format(index))
    np.save(mel_filename, mels, allow_pickle=False)

    if log_dir is not None:
      #save wav (mel -> wav)
      wav = audio.inv_mel_spectrogram(mels.T)
      audio.save_wav(wav, os.path.join(log_dir, 'wavs/speech-wav-{:05d}-mel.wav'.format(index)))
      if hparams.predict_linear:
        #save wav (linear -> wav)
        wav = audio.inv_linear_spectrogram(linear.T)
        audio.save_wav(wav, os.path.join(log_dir, 'wavs/speech-wav-{:05d}-linear.wav'.format(index)))
      plot.plot_alignment(alignment, os.path.join(log_dir, 'plots/speech-alignment-{:05d}.png'.format(index)),
        info='{}'.format(text), split_title=True)
      #save mel spectrogram plot
      plot.plot_spectrogram(mels, os.path.join(log_dir, 'plots/speech-mel-{:05d}.png'.format(index)),
        info='{}'.format(text), split_title=True)

    return mel_filename