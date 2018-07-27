import tensorflow as tf
import numpy as np

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
  # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
  # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
  cleaners='english_cleaners',

  # Audio:
  num_mels=80,
  num_freq=1025,
  sample_rate=25050,
  frame_length_ms=50,
  frame_shift_ms=12.5,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,
  hop_size = 256,
  predict_linear = True,
  win_size = None, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft)
  n_fft = 1024, #Extra window size is filled with 0 paddings to match this parameter

  rescale = True, #Whether to rescale audio prior to preprocessing
  rescaling_max = 0.999, #Rescaling value
  trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
  trim_fft_size = 512,
  trim_hop_size = 128,
  trim_top_db = 60,
  max_mel_frames = 900,  #Only relevant when clip_mels_length = True

  #note: might need to get rid of these
  signal_normalization = True,
  symmetric_mels = True, #Whether to scale the data to be symmetric around 0
  max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] 

  use_lws=True,
  silence_threshold=2,
  mask_decoder = True, #Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)

  # Model:
  outputs_per_step=2,
  embed_depth=256,
  prenet_depths=[256, 128],
  encoder_depth=256,
  rnn_depth=256,

  # Attention
  attention_depth=256,

  # Training:
  gst_test_size = None, #% of data to keep as test data, if None, tacotron_test_batches must be not None
  gst_test_batches = 32, #number of test batches (For Ljspeech: 10% ~= 41 batches of 32 samples)
  gst_data_random_state=1234, #random state for train test split repeatability
  random_seed = 5339,
  batch_size=32,
  adam_beta1=0.9,
  adam_beta2=0.999,
  initial_learning_rate=0.002,
  decay_learning_rate=True,
  use_cmudict=True,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes
  gst_scale_regularization = True, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)
  gst_reg_weight = 1e-6, 
  # Eval:
  max_iters=10000,
  griffin_lim_iters=60,
  power=1.5,              # Power to raise magnitudes to prior to Griffin-Lim

  #Global style token
  use_gst=True,     # When false, the scripit will do as the paper  "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron"
  num_gst=10,
  num_heads=4,       # Head number for multi-head attention
  style_embed_depth=256,
  reference_filters=[32, 32, 64, 64, 128, 128],
  reference_depth=128,
  style_att_type="mlp_attention", # Attention type for style attention module (dot_attention, mlp_attention)
  style_att_dim=128,

  #Wavenet Training
  wavenet_random_seed = 5339, # S=5, E=3, D=9 :)
  wavenet_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)
  wavenet_batch_size = 4, #batch size used to train wavenet.
  wavenet_test_size = 0.0441, #% of data to keep as test data, if None, wavenet_test_batches must be not None
  wavenet_test_batches = None, #number of test batches.
  wavenet_data_random_state = 1234, #random state for train test split repeatability

  wavenet_learning_rate = 1e-4,
  wavenet_adam_beta1 = 0.9,
  wavenet_adam_beta2 = 0.999,
  wavenet_adam_epsilon = 1e-6,

  wavenet_ema_decay = 0.9999, #decay rate of exponential moving average

  wavenet_dropout = 0.05, #drop rate of wavenet layers
  train_with_GTA = True, #Whether to use GTA mels to train WaveNet instead of ground truth mels.

  input_type="raw",
  quantize_channels=65536,  # 65536 (16-bit) (raw) or 256 (8-bit) (mulaw or mulaw-quantize) // number of classes = 256 <=> mu = 255

  log_scale_min=float(np.log(1e-14)), #Mixture of logistic distributions minimal log scale

  out_channels = 10 * 3, #This should be equal to quantize channels when input type is 'mulaw-quantize' else: num_distributions * 3 (prob, mean, log_scale)
  layers = 24, #Number of dilated convolutions (Default: Simplified Wavenet of Tacotron-2 paper)
  stacks = 4, #Number of dilated convolution stacks (Default: Simplified Wavenet of Tacotron-2 paper)
  residual_channels = 512,
  gate_channels = 512, #split in 2 in gated convolutions
  skip_out_channels = 256,
  kernel_size = 3,

  cin_channels = 80, #Set this to -1 to disable local conditioning, else it must be equal to num_mels!!
  upsample_conditional_features = True, #Whether to repeat conditional features or upsample them (The latter is recommended)
  upsample_scales = [16, 16], #prod(scales) should be equal to hop size
  freq_axis_kernel_size = 3,

  gin_channels = -1, #Set this to -1 to disable global conditioning, Only used for multi speaker dataset
  use_bias = True, #Whether to use bias in convolutional layers of the Wavenet

  max_time_sec = None,
  max_time_steps = 13000, #Max time steps in audio used to train wavenet (decrease to save memory)
  #Eval sentences (if no eval file was specified, these sentences are used for eval)
  sentences = [
  # From July 8, 2017 New York Times:
  'Scientists at the CERN laboratory say they have discovered a new particle.',
  'There\'s a way to measure the acute emotional intelligence that has never gone out of style.',
  'President Trump met with other leaders at the Group of 20 conference.',
  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # From Google's Tacotron example page:
  'Generative adversarial network or variational auto-encoder.',
  'Basilar membrane and otolaryngology are not auto-correlations.',
  'He has read the whole thing.',
  'He reads books.',
  "Don't desert me here in the desert!",
  'He thought it was time to present the present.',
  'Thisss isrealy awhsome.',
  'Punctuation sensitivity, is working.',
  'Punctuation sensitivity is working.',
  "The buses aren't the problem, they actually provide a solution.",
  "The buses aren't the PROBLEM, they actually provide a SOLUTION.",
  "The quick brown fox jumps over the lazy dog.",
  "does the quick brown fox jump over the lazy dog?",
  "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
  "She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
  "The blue lagoon is a nineteen eighty American romance adventure film.",
  "Tajima Airport serves Toyooka.",
  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
  #From Training data:
  'the rest being provided with barrack beds, and in dimensions varying from thirty feet by fifteen to fifteen feet by ten.',
  'in giltspur street compter, where he was first lodged.',
  'a man named burnett came with his wife and took up his residence at whitchurch, hampshire, at no great distance from laverstock,',
  'it appears that oswald had only one caller in response to all of his fpcc activities,',
  'he relied on the absence of the strychnia.',
  'scoggins thought it was lighter.',
  '''would, it is probable, have eventually overcome the reluctance of some of the prisoners at least, 
  and would have possessed so much moral dignity''',
  '''Sequence to sequence models have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization. 
  This project covers a sequence to sequence model trained to predict a speech representation from an input sequence of characters. We show that 
  the adopted architecture is able to perform this task with wild success.''',
  'Thank you so much for your support!',
  ]

  
)


def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
  return 'Hyperparameters:\n' + '\n'.join(hp)
