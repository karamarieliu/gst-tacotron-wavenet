import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import preprocessor
from hparams import hparams

def preprocess(args, input_folders, out_dir):
  mel_dir = os.path.join(out_dir, 'mels')
  wav_dir = os.path.join(out_dir, 'audio')
  linear_dir = os.path.join(out_dir, 'linear')
  os.makedirs(mel_dir, exist_ok=True)
  os.makedirs(wav_dir, exist_ok=True)
  os.makedirs(linear_dir, exist_ok=True)
  metadata = preprocessor.build_from_path(input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
  write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  mel_frames = sum([int(m[4]) for m in metadata])
  timesteps = sum([int(m[3]) for m in metadata])
  sr = hparams.sample_rate
  hours = timesteps / sr / 3600
  print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
    len(metadata), mel_frames, timesteps, hours))
  print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
  print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
  print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

def run_preprocess(args):
  input_folders = [os.path.join(args.base_dir, args.dataset)]
  output_folder = os.path.join(args.base_dir, args.output)
  preprocess(args, input_folders, output_folder)

def main():
  print('Initializing preprocessing..')
  supported_datasets = ['VCTK', 'LJSpeech-1.1']
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.getcwd())
  parser.add_argument('--output', default='training')
  parser.add_argument('--dataset', required=True, choices=supported_datasets)
  parser.add_argument('--n_jobs', type=int, default=cpu_count())
  args = parser.parse_args()
  
  if args.dataset not in supported_datasets:
    raise ValueError('Dataset value entered {} does not belong to supported datasets: {}'.format(
      args.dataset, supported_datasets))

  run_preprocess(args)
 

if __name__ == "__main__":
  main()
