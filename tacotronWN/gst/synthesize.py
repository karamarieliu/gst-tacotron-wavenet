import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from gst.synthesizer import Synthesizer
from util import audio
from time import sleep
from util.infolog import log
from tqdm import tqdm
import tensorflow as tf
import time
from text import cmudict
import random

def generate_fast(model, text):
	model.synthesize(text, None, None, None, None)

def run_live(args, checkpoint_path):
	#Log to Terminal without keeping any records in files
	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	#Generate fast greeting message
	greetings = 'Hello, Welcome to the Live testing tool. Please type a message and I will try to read it!'
	log(greetings)
	generate_fast(synth, greetings)

	#Interaction loop
	while True:
		try:
			text = input()
			generate_fast(synth, text)

		except KeyboardInterrupt:
			leave = 'Thank you for testing our features. see you soon.'
			log(leave)
			generate_fast(synth, leave)
			sleep(2)
			break

def get_output_base_path(checkpoint_path):
	base_dir = os.path.dirname(checkpoint_path)
	m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
	name = 'eval-%d' % int(m.group(1)) if m else 'eval'
	return os.path.join(base_dir, name)

#notes: when incorporating Rayhane's code from tacotron/synthesize,
#I only used run_synthesis as seen in run_eval below. No
#live modes or Rayhane's run_eval was used

def run_synthesis(args, checkpoint_path, output_dir):
	
	_p_cmudict = 0.5
	GTA = (args.GTA == 'True')
	if GTA:
		synth_dir = os.path.join(output_dir, 'gta')

		#Create output path if it doesn't exist
		os.makedirs(synth_dir, exist_ok=True)
	else:
		synth_dir = os.path.join(output_dir, 'natural')

		#Create output path if it doesn't exist
		os.makedirs(synth_dir, exist_ok=True)
	metadata_filename = os.path.join(args.input_dir, 'train.txt')

	if hparams.use_cmudict:
			cmudict_path = os.path.join(os.path.dirname(metadata_filename), 'cmudict-0.7b')
			if not os.path.isfile(cmudict_path):
				raise Exception('If use_cmudict=True, you must download cmu dictionary first. ' +
					'Run shell as:\n wget -P %s http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b'  % self._datadir)
			_cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=False)
			log('Loaded CMUDict with %d unambiguous entries' % len(_cmudict))
	else:
			_cmudict = None

	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, gta=GTA)
	with open(metadata_filename, encoding='utf-8') as f:
		metadata = [line.strip().split('|') for line in f]

	log('starting synthesis')
	mel_dir = os.path.join(args.input_dir, 'mels')
	wav_dir = os.path.join(args.input_dir, 'audio')
		
	with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
		for i, meta in enumerate(tqdm(metadata)):

			_punctuation_re = re.compile(r'([\.,"\-_:]+)')
			text =  re.sub(_punctuation_re, r' \1 ', meta[3])
			if _cmudict and random.random() < _p_cmudict:
				text = ' '.join([maybe_get_arpabet(_cmudict, word) for word in text.split(' ')])
			mel_filename = os.path.join(mel_dir, meta[1])
			wav_filename = os.path.join(wav_dir, meta[0])
			mel_output_filename = synth.synthesize(text, i+1, synth_dir, None, mel_filename)
			file.write('{}|{}|{}|{}\n'.format(wav_filename, mel_filename, mel_output_filename, text))
	log('synthesized mel spectrograms at {}'.format(synth_dir))
	return os.path.join(synth_dir, 'map.txt')

def maybe_get_arpabet(cmudict, word):
		arpabet = cmudict.lookup(word)
		return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word
#IS THIS ONE BETTER???
def old_run_synth():	
	print(hparams_debug_string())
	GTA = False
	mel_targets = args.mel_targets
	reference_mel = None
	if args.mel_targets is not None:
		GTA = True
		mel_targets = np.load(args.mel_targets)
	synth = Synthesizer(teacher_forcing_generating=GTA)
	synth.load(args.checkpoint, args.reference_audio)
	base_path = get_output_base_path(args.checkpoint)

	with open(path, 'wb') as f:
		print('Synthesizing: %s' % args.text)
		print('Output wav file: %s' % path)
		print('Output alignments: %s' % alignment_path)
		f.write(synth.synthesize(args.text, mel_targets=mel_targets, reference_mel=reference_mel, alignment_path=alignment_path))

def run_eval(args, checkpoint_path, output_dir, sentences, reference_mel):
			eval_dir = os.path.join(output_dir, 'eval')
			log_dir = os.path.join(output_dir, 'logs-eval')

			assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir) #mels_dir = wavenet_input_dir

			#Create output path if it doesn't exist
			os.makedirs(eval_dir, exist_ok=True)
			os.makedirs(log_dir, exist_ok=True)
			os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
			os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
			print(sentences)
			log(hparams_debug_string())
			synth = Synthesizer()
			synth.load(checkpoint_path, reference_mel=reference_mel)
			with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
				for i, text in enumerate(tqdm(sentences)):
					start = time.time()
					mel_filename = synth.synthesize(text, i+1, eval_dir, log_dir, None, reference_mel=reference_mel)

					file.write('{}|{}\n'.format(text, mel_filename))
			log('synthesized mel spectrograms at {}'.format(eval_dir))
			return eval_dir

def gst_synthesize(args, checkpoint, sentences=None, reference_mel=None):
	output_dir = "gst_" + args.output_dir
	checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
	
	log('loaded model at {}'.format(checkpoint_path))
 
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	hparams.parse(args.hparams)
	if args.mode == 'eval':
		return run_eval(args, checkpoint_path, output_dir, sentences, reference_mel)
	elif args.mode == 'synthesis':
		return run_synthesis(args, checkpoint_path, output_dir)
	else:
		run_live(args, checkpoint_path)

