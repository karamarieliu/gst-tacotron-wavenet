import librosa
import librosa.filters
import math
import numpy as np
import tensorflow as tf
from scipy import signal
from hparams import hparams
from scipy.io import wavfile
import lws
import datetime

def load_wav(path):
  return librosa.core.load(path, sr=hparams.sample_rate)[0]

def trim_silence(wav):
  '''Trim leading and trailing silence
  Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
  '''
  #Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
  return librosa.effects.trim(wav, top_db= hparams.trim_top_db, frame_length=hparams.trim_fft_size, hop_length=hparams.trim_hop_size)[0]

def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  librosa.output.write_wav(path, wav.astype(np.float32), hparams.sample_rate)


def preemphasis(x):
  return signal.lfilter([1, -hparams.preemphasis], [1], x)

def start_and_end_indices(quantized, silence_threshold=2):
  for start in range(quantized.size):
          if abs(quantized[start] - 127) > silence_threshold:
                  break
  for end in range(quantized.size - 1, 1, -1):
          if abs(quantized[end] - 127) > silence_threshold:
                  break

  assert abs(quantized[start] - 127) > silence_threshold
  assert abs(quantized[end] - 127) > silence_threshold

  return start, end
      
def inv_preemphasis(x):
  return signal.lfilter([1], [1, -hparams.preemphasis], x)


def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
  return _normalize(S)

def linearspectrogram(wav):
  D = _stft(wav)
  S = _amp_to_db(np.abs(D)) - hparams.ref_level_db

  if hparams.signal_normalization:
    return _normalize(S)
  return S

def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hparams.power))          # Reconstruct phase


def inv_spectrogram_tensorflow(spectrogram):
  '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

  Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
  inv_preemphasis on the output after running the graph.
  '''
  S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hparams.ref_level_db)
  return _griffin_lim_tensorflow(tf.pow(S, hparams.power))


def inv_mel_spectrogram(mel_spectrogram):
  '''Converts mel spectrogram to waveform using librosa'''
  if hparams.signal_normalization:
    D = _denormalize(mel_spectrogram)
  else:
    D = mel_spectrogram
  S = (_db_to_amp(D + hparams.ref_level_db))  # Convert back to linear
  S = _mel_to_linear(S)
  if hparams.use_lws:
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
    y = processor.istft(D).astype(np.float32)
    return y
  else:
    return _griffin_lim(S ** hparams.power)


def inv_mel_spectrogram_tensorflow(mel_spectrogram):
  '''Converts mel spectrogram to waveform using librosa'''
  if hparams.signal_normalization is not None:
    D = _denormalize_tensorflow(mel_spectrogram)
  else:
    D = mel_spectrogram

  S = _mel_to_linear_tensorflow(_db_to_amp_tensorflow(D + hparams.ref_level_db))  # Convert back to linear

  if hparams.use_lws is not None:
    sess = tf.Session()
    processor = _lws_processor()
    with sess.as_default():
      D = processor.run_lws((tf.pow(tf.transpose(S),hparams.power)).eval())
      y = processor.istft(D).astype(np.float32)
      return y
  else:
    return _griffin_lim_tensorflow(tf.pow(S, hparams.power))


def inv_linear_spectrogram(linear_spectrogram):
  '''Converts linear spectrogram to waveform using librosa'''
  if hparams.signal_normalization:
    D = _denormalize(linear_spectrogram)
  else:
    D = linear_spectrogram

  S = _db_to_amp(D + hparams.ref_level_db) #Convert back to linear

  if hparams.use_lws:
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
    y = processor.istft(D).astype(np.float32)
    return y
  else:
    return _griffin_lim(S ** hparams.power)
    
def inv_linear_spectrogram_tensorflow(linear_spectrogram):
  '''Converts linear spectrogram to waveform using tensorflow'''
  if hparams.signal_normalization:
    D = _denormalize_tensorflow(linear_spectrogram)
  else:
    D = linear_spectrogram

  S = _db_to_amp_tensorflow(D + hparams.ref_level_db) #Convert back to linear

  if hparams.use_lws:
    sess = tf.Session()
    processor = _lws_processor()
    with sess.as_default():
      D = processor.run_lws((tf.pow(tf.transpose(S),hparams.power)).eval())
      y = processor.istft(D).astype(np.float32)
      return y
  else:
    return _griffin_lim_tensorflow(tf.pow(S, hparams.power))

def _lws_processor():
  return lws.lws(hparams.n_fft, get_hop_size(), fftsize=hparams.win_size, mode="speech")


def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hparams.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)

def get_hop_size():
  hop_size = hparams.hop_size
  if hop_size is None:
          assert hparams.frame_shift_ms is not None
          hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
  return hop_size


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(hparams.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


def _griffin_lim_tensorflow(S):
  '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
  with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(hparams.griffin_lim_iters):
      est = _stft_tensorflow(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)


def _stft(y):
  return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(), win_length=hparams.win_size)


def _istft(y):
  return librosa.istft(y, hop_length=get_hop_size(), win_length=hparams.win_size)


def _stft_tensorflow(signals):
  return tf.contrib.signal.stft(signals, hparams.win_size, get_hop_size(), hparams.n_fft, pad_end=False)


def _istft_tensorflow(stfts):
  return tf.contrib.signal.inverse_stft(stfts, hparams.win_size, get_hop_size(), hparams.n_fft)

def num_frames(length, fsize, fshift):
  """Compute number of time frames of spectrogram
  """
  pad = (fsize - fshift)
  if length % fshift == 0:
          M = (length + pad * 2 - fsize) // fshift + 1
  else:
          M = (length + pad * 2 - fsize) // fshift + 2
  return M


def pad_lr(x, fsize, fshift):
  """Compute left and right padding
  """
  M = num_frames(len(x), fsize, fshift)
  pad = (fsize - fshift)
  T = len(x) + 2 * pad
  r = (M - 1) * fshift + fsize - T
  return pad, pad + r

# Conversions:

_mel_basis = None
_inv_mel_basis = None
_mel_basis_tf = []
_inv_mel_basis_tf = []

def _mel_to_linear(mel_spectrogram):
  global _inv_mel_basis
  if _inv_mel_basis is None:
    _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
  return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _mel_to_linear_tensorflow(mel_spectrogram):
  global _inv_mel_basis_tf
  if _inv_mel_basis_tf ==[]:
    _inv_mel_basis_tf = np.linalg.pinv(_build_mel_basis())
  inv_mel_basis_tf = tf.convert_to_tensor(_inv_mel_basis_tf)
  mel_spectrogram = tf.cast(mel_spectrogram, tf.float64)
  return tf.maximum(tf.convert_to_tensor(1e-10, tf.float64), tf.matmul(_inv_mel_basis_tf, mel_spectrogram))

def _linear_to_mel_tensorflow(spectrogram):
  global _mel_basis_tf
  if _mel_basis_tf == []:
    _mel_basis_tf = _build_mel_basis()
  mel_basis_tf = tf.convert_to_tensor(_mel_basis_tf)
  spectrogram = tf.cast(spectrogram, tf.float64)
  return tf.matmul(_mel_basis_tf, spectrogram)

def _build_mel_basis():
  n_fft = (hparams.num_freq - 1) * 2
  return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(S):
  return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

def _denormalize_tensorflow(S):
  return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
