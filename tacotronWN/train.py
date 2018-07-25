import argparse
import tensorflow as tf 
from gst.train import gst_train
from wavenet_vocoder.train import wavenet_train
from gst.synthesize import gst_synthesize
from hparams import hparams
import os
from util import infolog
import traceback

from time import sleep

log = infolog.log


def save_seq(file, sequence, input_path):
	'''Save Tacotron-2 training state to disk. (To skip for future runs)
	'''
	sequence = [str(int(s)) for s in sequence] + [input_path]
	with open(file, 'w') as f:
		f.write('|'.join(sequence))

def read_seq(file, restore):
	'''Load Tacotron-2 training state from disk. (To skip if not first run)
	'''
	if os.path.isfile(file) and restore==True:
		with open(file, 'r') as f:
			sequence = f.read().split('|')

		return [bool(int(s)) for s in sequence[:-1]], sequence[-1]
	else:
		return [0, 0, 0], ''

def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
	log_dir = os.path.join(args.base_dir, 'logs-{}'.format(args.name))
	os.makedirs(log_dir, exist_ok=True)
	infolog.init(os.path.join(log_dir, 'Terminal_train_log'), args.name)
	return log_dir, modified_hp

def train(args, log_dir, hparams):
	state_file = os.path.join(log_dir, 'state_log')
	#Get training states
	(gst_state, GTA_state, wave_state), input_path = read_seq(state_file, args.restore_step)
	try:

		if not gst_state:
			log('\n#############################################################\n')
			log('GST Train\n')
			log('###########################################################\n')
			checkpoint = gst_train(log_dir, args)
			tf.reset_default_graph()
			#Sleep 1 second to let previous graph close and avoid error messages while synthesis
			sleep(1)
			if checkpoint is None:
				raise('Error occured while training Tacotron, Exiting!')
			gst_state = 1
			save_seq(state_file, [gst_state, GTA_state, wave_state], input_path)

		if not GTA_state:
			log('\n#############################################################\n')
			log('GST GTA Synthesis\n')
			log('###########################################################\n')
			input_path = gst_synthesize(args, checkpoint)
			GTA_state = 1
			save_seq(state_file, [gst_state, GTA_state, wave_state], input_path)
		
		if not wave_state:
			log('\n#############################################################\n')
			log('Wavenet Train\n')
			log('###########################################################\n')
			print("training the wave!!!!!!!!!!")
			checkpoint = wavenet_train(args, log_dir, hparams, "gst_output/gta/map.txt")
			if checkpoint is None:
				raise ('Error occured while training Wavenet, Exiting!')
			wave_state = 1
			save_seq(state_file, [gst_state, GTA_state, wave_state], input_path)
		if input_path == '' or input_path is None:
			raise RuntimeError('input_path has an unpleasant value -> {}'.format(input_path))

		

		if wave_state and GTA_state and gst_state:
			log('TRAINING IS ALREADY COMPLETE!!')
	
	except (Exception, OSError, NameError, ValueError, AttributeError) as e:
		log('Exiting due to error: %s' % e, slack=True)
		traceback.print_exc()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--model', default='tacotron')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--name', default='dir', help='Name of logging directory.')
	parser.add_argument('--gst_input', default='training/train.txt')
	parser.add_argument('--wavenet_input', default='gst_output/gta/map.txt')
	parser.add_argument('--input_dir', default='training/', help='folder to contain inputs sentences/targets')
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
	parser.add_argument('--restore_step', type=bool, default=False, help='Set this to True to restore training')
	parser.add_argument('--summary_interval', type=int, default=500,
		help='Steps between running summary ops')
	parser.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training')
	parser.add_argument('--checkpoint_interval', type=int, default=2000,
		help='Steps between writing checkpoints')
	parser.add_argument('--eval_interval', type=int, default=10000,
		help='Steps between eval on test data')
	parser.add_argument('--gst_train_steps', type=int, default=90000, help='total number of tacotron training steps')
	parser.add_argument('--wavenet_train_steps', type=int, default=200000, help='total number of wavenet training steps')
	parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level')
	parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
	args = parser.parse_args()
	log_dir, hparams = prepare_run(args)
	train(args, log_dir, hparams)

if __name__ == '__main__':
	main()
