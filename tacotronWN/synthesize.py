import argparse
from gst.synthesize import gst_synthesize
from wavenet_vocoder.synthesize import wavenet_synthesize
from util.infolog import log
from hparams import hparams
from warnings import warn
import os
from util import audio
import numpy as np


def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	run_name = args.name or args.gst_name 
	gst_checkpoint = os.path.join('logs-' + run_name, 'gst_' + args.checkpoint)
	run_name = args.name or args.wavenet_name
	wave_checkpoint = os.path.join('logs-' + run_name, 'wave_' + args.checkpoint)
	return gst_checkpoint, wave_checkpoint, modified_hp

def get_sentences(args):
	if args.text != '':
		sentences = args.text
	else:
		sentences = hparams.sentences
	return sentences

def synthesize(args, hparams, gst_checkpoint, wave_checkpoint, sentences, reference_mel):
	log('Running End-to-End TTS Evaluation. Model: {}'.format(args.name))
	log('Synthesizing mel-spectrograms from text..')
	wavenet_in_dir = gst_synthesize(args, gst_checkpoint, sentences, reference_mel)
	log('Synthesizing audio from mel-spectrograms.. (This may take a while)')
	wavenet_synthesize(args, hparams, wave_checkpoint)
	log('Tacotron-2 TTS synthesis complete!')



def main():
	accepted_modes = ['eval', 'synthesis', 'live']
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--name', required = True, help='Name of logging directory.')
	parser.add_argument('--mels_dir', default='gst_output/eval/', help='folder to contain mels to synthesize audio from using the Wavenet')
	parser.add_argument('--mode', default='eval', help='mode of run: can be one of {}'.format(accepted_modes))
	parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
	parser.add_argument('--text', required=True, default=None, help='Single test text sentence')
	parser.add_argument('--reference_audio', default=None, help='Reference audio path')
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')

	
	args = parser.parse_args()
	

	if args.mode not in accepted_modes:
		raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))

	if args.mode=='live' and args.model=='Wavenet':
		raise RuntimeError('Wavenet vocoder cannot be tested live due to its slow generation. Live only works with Tacotron!')

	if args.GTA not in ('True', 'False'):
		raise ValueError('GTA option must be either True or False')

	if args.mode == 'live':
		warn('Requested a live evaluation with Tacotron-2, Wavenet will not be used!')
	if args.mode == 'synthesis':
		raise ValueError('I don\'t recommend running WaveNet on entire dataset.. The world might end before the synthesis :) (only eval allowed)')

	gst_checkpoint, wave_checkpoint, hparams = prepare_run(args)
	sentences = get_sentences(args)
	if args.reference_audio is not None:
		ref_wav = audio.load_wav(args.reference_audio)
		reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
	else:
		reference_mel = None
	
	synthesize(args, hparams, gst_checkpoint, wave_checkpoint, sentences, reference_mel)
	

if __name__ == '__main__':
	main()