import numpy as np
import os
import re
import random
import tensorflow as tf
import threading
import time
import traceback
from text import cmudict, text_to_sequence
from util.infolog import log
from sklearn.model_selection import train_test_split
from hparams import hparams

_batches_per_group = 32
_p_cmudict = 0.5
_pad=0

class DataFeeder(threading.Thread):
	'''Feeds batches of data into a queue on a background thread.'''

	def __init__(self, coordinator, metadata_filename, hparams):
		super(DataFeeder, self).__init__()
		self._coord = coordinator
		self._hparams = hparams
		self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		self.train_offset = 0
		self.test_offset = 0


		# Load metadata:
		self._mel_dir = os.path.join(os.path.dirname(metadata_filename), 'mels')
		self._datadir = os.path.dirname(metadata_filename)
		self._linear_dir = os.path.join(os.path.dirname(metadata_filename), 'linear')		
		with open(metadata_filename, encoding='utf-8') as f:
			self._metadata = [line.strip().split('|') for line in f]
			frame_shift_ms = hparams.hop_size / hparams.sample_rate
			hours = sum((int(x[4]) for x in self._metadata)) * frame_shift_ms / 3600
			log('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))

		#Train test split
		if hparams.gst_test_size is None:
			assert hparams.gst_test_batches is not None

		test_size = (hparams.gst_test_size if hparams.gst_test_size is not None 
			else hparams.gst_test_batches * hparams.batch_size)
		indices = np.arange(len(self._metadata))
		train_indices, test_indices = train_test_split(indices,
			test_size=test_size, random_state=hparams.gst_data_random_state)

		#Make sure test_indices is a multiple of batch_size else round up
		len_test_indices = _round_up(len(test_indices), hparams.batch_size)
		extra_test = test_indices[len_test_indices:]
		test_indices = test_indices[:len_test_indices]
		train_indices = np.concatenate([train_indices, extra_test])

		self._train_meta = list(np.array(self._metadata)[train_indices])
		self._test_meta = list(np.array(self._metadata)[test_indices])

		self.test_steps = len(self._test_meta) // hparams.batch_size

		if hparams.gst_test_size is None:
			assert hparams.gst_test_batches == self.test_steps

		# Create placeholders for inputs and targets. Don't specify batch size because we want to
		# be able to feed different sized batches at eval time.
		self._placeholders = [
			tf.placeholder(tf.int32, [None, None], 'inputs'),
			tf.placeholder(tf.int32, [None], 'input_lengths'),
			tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
			tf.placeholder(tf.float32, shape=(None, None), name='token_targets'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_freq), name='linear_targets'),
			tf.placeholder(tf.int32, shape=(None, ), name='targets_lengths'),    
			]

		# Create queue for buffering data:
		queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32], name='input_queue')
		self._enqueue_op = queue.enqueue(self._placeholders)
		self.inputs, self.input_lengths, self.mel_targets, self.token_targets, self.linear_targets, self.targets_lengths = queue.dequeue()

		self.inputs.set_shape(self._placeholders[0].shape)
		self.input_lengths.set_shape(self._placeholders[1].shape)
		self.mel_targets.set_shape(self._placeholders[2].shape)
		self.token_targets.set_shape(self._placeholders[3].shape)
		self.linear_targets.set_shape(self._placeholders[4].shape)
		self.targets_lengths.set_shape(self._placeholders[5].shape)

		# Create eval queue for buffering eval data
		eval_queue = tf.FIFOQueue(1, [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32], name='eval_queue')
		self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
		self.eval_inputs, self.eval_input_lengths, self.eval_mel_targets, self.eval_token_targets, \
			self.eval_linear_targets, self.eval_targets_lengths = eval_queue.dequeue()

		self.eval_inputs.set_shape(self._placeholders[0].shape)
		self.eval_input_lengths.set_shape(self._placeholders[1].shape)
		self.eval_mel_targets.set_shape(self._placeholders[2].shape)
		self.eval_token_targets.set_shape(self._placeholders[3].shape)
		self.eval_linear_targets.set_shape(self._placeholders[4].shape)
		self.eval_targets_lengths.set_shape(self._placeholders[5].shape)

		# Load CMUDict: If enabled, this will randomly substitute some words in the training data with
		# their ARPABet equivalents, which will allow you to also pass ARPABet to the model for
		# synthesis (useful for proper nouns, etc.)
		if hparams.use_cmudict:
			cmudict_path = os.path.join(os.path.dirname(metadata_filename), 'cmudict-0.7b')
			if not os.path.isfile(cmudict_path):
				raise Exception('If use_cmudict=True, you must download cmu dictionary first. ' +
					'Run shell as:\n wget -P %s http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b'  % self._datadir)
			self._cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=False)
			log('Loaded CMUDict with %d unambiguous entries' % len(self._cmudict))
		else:
			self._cmudict = None

		

	def start_in_session(self, session):
		self._session = session
		thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

		thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()


	def _enqueue_next_train_group(self):
		while not self._coord.should_stop():
			start = time.time()

			# Read a group of examples:
			n = self._hparams.batch_size
			r = self._hparams.outputs_per_step
			examples = [self._get_next_example() for i in range(n * _batches_per_group)]

			# Bucket examples based on similar output sequence length for efficiency:
			examples.sort(key=lambda x: x[-1])
			batches = [examples[i:i+n] for i in range(0, len(examples), n)]
			random.shuffle(batches)

			log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
			for batch in batches:
				feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
				self._session.run(self._enqueue_op, feed_dict=feed_dict)


	def _get_next_example(self):
		'''Loads a single example (input, mel_target, linear_target, cost) from disk'''
		if self.train_offset >= len(self._train_meta):
			self.train_offset = 0
			random.shuffle(self._train_meta)
		meta = self._train_meta[self.train_offset]
		self.train_offset += 1

		_punctuation_re = re.compile(r'([\.,"\-_:]+)')
		text =  re.sub(_punctuation_re, r' \1 ', meta[3])
		if self._cmudict and random.random() < _p_cmudict:
			text = ' '.join([self._maybe_get_arpabet(word) for word in text.split(' ')])

		mel_target = np.load(os.path.join(self._mel_dir, meta[1]))
		token_target = np.asarray([0.] * (len(mel_target) - 1))
		input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
		linear_target = np.load(os.path.join(self._linear_dir, meta[2]))
		return (input_data, mel_target, token_target, linear_target, len(linear_target))
	

	
	def _get_test_groups(self):
		meta = self._test_meta[self.test_offset]
		self.test_offset += 1
		_punctuation_re = re.compile(r'([\.,"\-_:]+)')
		text =  re.sub(_punctuation_re, r' \1 ', meta[5])
		if self._cmudict and random.random() < _p_cmudict:
			text = ' '.join([self._maybe_get_arpabet(word) for word in text.split(' ')])

		input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
		mel_target = np.load(os.path.join(self._mel_dir, meta[1]))
		#Create parallel sequences containing zeros to represent a non finished sequence
		token_target = np.asarray([0.] * (len(mel_target) - 1))
		linear_target = np.load(os.path.join(self._linear_dir, meta[2]))
		return (input_data, mel_target, token_target, linear_target, len(mel_target))


	def make_test_batches(self):
		start = time.time()

		# Read a group of examples
		n = self._hparams.batch_size
		r = self._hparams.outputs_per_step

		#Test on entire test set
		examples = [self._get_test_groups() for i in range(len(self._test_meta))]

		# Bucket examples based on similar output sequence length for efficiency
		examples.sort(key=lambda x: x[-1])
		batches = [examples[i: i+n] for i in range(0, len(examples), n)]
		np.random.shuffle(batches)

		log('\nGenerated {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
		return batches, r


	def _enqueue_next_test_group(self):
		#Create test batches once and evaluate on them for all test steps
		test_batches, r = self.make_test_batches()
		while not self._coord.should_stop():
			for batch in test_batches:
				feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
				self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

	def _maybe_get_arpabet(self, word):
		arpabet = self._cmudict.lookup(word)
		return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word

#TODO: figure out what token and target len is here!!!!!!!!!!!!!!!

def _prepare_batch(batch, outputs_per_step):
		random.shuffle(batch)
		inputs = _prepare_inputs([x[0] for x in batch])
		input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
		mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
		token_targets=_prepare_token_targets([x[2] for x in batch], outputs_per_step)
		linear_targets = _prepare_targets([x[3] for x in batch], outputs_per_step)
		targets_lengths = np.asarray([x[-1] for x in batch], dtype=np.int32) #Used to mask loss	
		return (inputs, input_lengths, mel_targets, token_targets, linear_targets, targets_lengths)



def _prepare_inputs(inputs):
	max_len = max((len(x) for x in inputs))
	return np.stack([_pad_input(x, max_len) for x in inputs])

def _prepare_token_targets(targets, alignment):
	max_len = max([len(t) for t in targets]) + 1
	return np.stack([_pad_token_target(t, _round_up(max_len, alignment)) for t in targets])

def _prepare_targets(targets, alignment):
	max_len = max((len(t) for t in targets))
	return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
	#print("\n Input \n")
	return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
	#explicitly setting the padding to a value that doesn't originally exist in the spectogram
	#to avoid any possible conflicts, without affecting the output range of the model too much
	if hparams.symmetric_mels is not None:
		_target_pad = -(hparams.max_abs_value + .1)
	else:
		_target_pad = -0.1
	#print("\n Pad \n")
	return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_target_pad)

def _pad_token_target(t, length):	
	#print("\n Token \n")
	return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=1.)

def _round_up(x, multiple):
	remainder = x % multiple
	return x if remainder == 0 else x + multiple - remainder