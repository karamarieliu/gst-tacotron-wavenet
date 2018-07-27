import tensorflow as tf
import traceback
import os
from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text
from util import audio, plot, ValueWindow, infolog
import time
import math
from tqdm import tqdm
from datetime import datetime
import numpy as np
log = infolog.log



def add_train_stats(model):
	with tf.variable_scope('stats') as scope:
		tf.summary.histogram('mel_outputs', model.mel_outputs)
		tf.summary.histogram('mel_targets', model.mel_targets)
		if hparams.predict_linear:
			tf.summary.scalar('linear_loss', model.linear_loss)
		tf.summary.scalar('regularization_loss', model.regularization_loss)
		#tf.summary.scalar('stop_token_loss', model.stop_token_loss)
		tf.summary.scalar('loss', model.loss)
		tf.summary.scalar('learning_rate', model.learning_rate) #Control learning rate decay speed
		gradient_norms = [tf.norm(grad) for grad in model.gradients]
		tf.summary.histogram('gradient_norm', gradient_norms)
		tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)) #visualize gradients (in case of explosion)
		return tf.summary.merge_all()

def add_eval_stats(summary_writer, step, linear_loss, loss):
	values = [
	tf.Summary.Value(tag='eval_model/eval_stats/eval_loss', simple_value=loss),]
	if linear_loss is not None:
		values.append(tf.Summary.Value(tag='model/eval_stats/eval_linear_loss', simple_value=linear_loss))
	test_summary = tf.Summary(value=values)
	summary_writer.add_summary(test_summary, step)


#TODO: take ou linear targ in else case	for both of these 
def model_train_mode(args, feeder, hparams, global_step):
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
		model = create_model(args.model, hparams)
		if hparams.predict_linear:
			model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, None, feeder.token_targets, linear_targets=feeder.linear_targets, 
				targets_lengths=feeder.targets_lengths, global_step=global_step,
				is_training=True)
		else:
			model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, None, feeder.token_targets, linear_targets=feeder.linear_targets,
				targets_lengths=feeder.targets_lengths, global_step=global_step,
				is_training=True)

		model.add_loss()
		model.add_optimizer(global_step)
		stats = add_train_stats(model)
		return model, stats

def model_test_mode(args, feeder, hparams, global_step):
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
		model = create_model(args.model, hparams)
		if hparams.predict_linear:
			model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets, None, feeder.eval_token_targets, 
				linear_targets=feeder.eval_linear_targets, targets_lengths=feeder.eval_targets_lengths, global_step=global_step,
				is_training=False, is_evaluating=True)
		else:
			model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets, None, feeder.eval_token_targets, 
				linear_targets=feeder.eval_linear_targets,targets_lengths=feeder.eval_targets_lengths, global_step=global_step, 
				is_training=False, is_evaluating=True)
		model.add_loss()
		return model

def time_string():
	return datetime.now().strftime('%Y-%m-%d %H:%M')

def get_git_commit():
	subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
	commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
	log('Git commit: %s' % commit)
	return commit

def gst_train(log_dir, args):
	commit = get_git_commit() if args.git else 'None'
	save_dir = os.path.join(log_dir, 'gst_pretrained/')
	checkpoint_path = os.path.join(save_dir, 'gst_model.ckpt')
	input_path = os.path.join(args.base_dir, args.gst_input)
	plot_dir = os.path.join(log_dir, 'plots')
	wav_dir = os.path.join(log_dir, 'wavs')
	mel_dir = os.path.join(log_dir, 'mel-spectrograms')
	eval_dir = os.path.join(log_dir, 'eval-dir')
	eval_plot_dir = os.path.join(eval_dir, 'plots')
	eval_wav_dir = os.path.join(eval_dir, 'wavs')
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(plot_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(eval_plot_dir, exist_ok=True)
	os.makedirs(eval_wav_dir, exist_ok=True)

	log('Checkpoint path: %s' % checkpoint_path)
	log('Loading training data from: %s' % input_path)
	log(hparams_debug_string())

	#Start by setting a seed for repeatability
	tf.set_random_seed(hparams.random_seed)
	
	# Set up DataFeeder:
	coord = tf.train.Coordinator()
	with tf.variable_scope('datafeeder') as scope:
		feeder = DataFeeder(coord, input_path, hparams)

	# Set up model:
	global_step = tf.Variable(0, name='global_step', trainable=False)
	model, stats = model_train_mode(args, feeder, hparams, global_step)
	eval_model = model_test_mode(args, feeder, hparams, global_step)
		
	# Bookkeeping:
	step = 0
	time_window = ValueWindow(100)
	loss_window = ValueWindow(100)
	saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

	#Memory allocation on the GPU as needed
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True


	# Train!
	with tf.Session(config=config) as sess:
		try:
			summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
			sess.run(tf.global_variables_initializer())
			checkpoint_state = False
			#saved model restoring
			if args.restore_step:
				#Restore saved model if the user requested it, Default = True.
				try:
					checkpoint_state = tf.train.get_checkpoint_state(save_dir)
				except tf.errors.OutOfRangeError as e:
					log('Cannot restore checkpoint: {}'.format(e))

			if (checkpoint_state and checkpoint_state.model_checkpoint_path):
				log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
				saver.restore(sess, checkpoint_state.model_checkpoint_path)

			else:
				if not args.restore_step:
					log('Starting new training!')
				else:
					log('No model to load at {}'.format(save_dir))

			feeder.start_in_session(sess)

			while not coord.should_stop() and step < args.gst_train_steps:
				start_time = time.time()
				step, loss, opt = sess.run([global_step, model.loss, model.optimize])
				time_window.append(time.time() - start_time)
				loss_window.append(loss)
				message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
					step, time_window.average, loss, loss_window.average)
				log(message, slack=(step % args.checkpoint_interval == 0))

				if loss > 100 or math.isnan(loss):
					log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
					raise Exception('Loss Exploded')

				if step % args.summary_interval == 0:
					log('Writing summary at step: %d' % step)
					summary_writer.add_summary(sess.run(stats), step)

				if step % args.eval_interval == 0:
					#Run eval and save eval stats
					log('\nRunning evaluation at step {}'.format(step))

					eval_losses = []
					linear_losses = []

					#TODO: FIX TO ENCOMPASS MORE LOSS
					for i in tqdm(range(feeder.test_steps)):
									eloss, linear_loss, mel_p, mel_t, t_len, align, lin_p = sess.run([eval_model.loss, eval_model.linear_loss, 
										eval_model.mel_outputs[0], eval_model.mel_targets[0], eval_model.targets_lengths[0], eval_model.alignments[0], 
										eval_model.linear_outputs[0]])
									eval_losses.append(eloss)
									linear_losses.append(linear_loss)


					eval_loss = sum(eval_losses) / len(eval_losses)
					linear_loss = sum(linear_losses) / len(linear_losses)

					wav = audio.inv_linear_spectrogram(lin_p.T)
					audio.save_wav(wav, os.path.join(eval_wav_dir, 'step-{}-eval-waveform-linear.wav'.format(step)))
					log('Saving eval log to {}..'.format(eval_dir))
					#Save some log to monitor model improvement on same unseen sequence

					wav = audio.inv_mel_spectrogram(mel_p.T)
					audio.save_wav(wav, os.path.join(eval_wav_dir, 'step-{}-eval-waveform-mel.wav'.format(step)))

					plot.plot_alignment(align, os.path.join(eval_plot_dir, 'step-{}-eval-align.png'.format(step)),
									info='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, eval_loss),
									max_len=t_len // hparams.outputs_per_step)
					plot.plot_spectrogram(mel_p, os.path.join(eval_plot_dir, 'step-{}-eval-mel-spectrogram.png'.format(step)),
									info='{}, {}, step={}, loss={:.5}'.format(args.model, time_string(), step, eval_loss), target_spectrogram=mel_t,
									)

					log('Eval loss for global step {}: {:.3f}'.format(step, eval_loss))
					log('Writing eval summary!')
					add_eval_stats(summary_writer, step, linear_loss, eval_loss)      

				if step % args.checkpoint_interval == 0 or step == args.gst_train_steps:
					log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
					saver.save(sess, checkpoint_path, global_step=step)
					log('Saving audio and alignment...')
					input_seq, mel_pred, alignment, target, target_len = sess.run([model.inputs[0],
						model.mel_outputs[0],
						model.alignments[0],
						model.mel_targets[0],
						model.targets_lengths[0],
						])
					
					#save predicted mel spectrogram to disk (debug)
					mel_filename = 'mel-prediction-step-{}.npy'.format(step)
					np.save(os.path.join(mel_dir, mel_filename), mel_pred.T, allow_pickle=False)

					#save griffin lim inverted wav for debug (mel -> wav)
					wav = audio.inv_mel_spectrogram(mel_pred.T)
					audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-mel.wav'.format(step)))

					#save alignment plot to disk (control purposes)
					plot.plot_alignment(alignment, os.path.join(plot_dir, 'step-{}-align.png'.format(step)),
									info='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss),
									max_len=target_len // hparams.outputs_per_step)
					#save real and predicted mel-spectrogram plot to disk (control purposes)
					plot.plot_spectrogram(mel_pred, os.path.join(plot_dir, 'step-{}-mel-spectrogram.png'.format(step)),
									info='{}, {}, step={}, loss={:.5}'.format(args.model, time_string(), step, loss), target_spectrogram=target,
									max_len=target_len)
					log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))

			log('GST Taco training complete after {} global steps!'.format(args.gst_train_steps))
			return save_dir

		
		except Exception as e:
			log('Exiting due to exception: %s' % e, slack=True)
			traceback.print_exc()
			coord.request_stop(e)

