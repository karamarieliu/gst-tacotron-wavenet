# Global Style Tokens w/ Tacotron and a WaveNet Vocoder 

Main structure of Tacotron w/ WaveNet forked from https://github.com/Rayhane-mamah
Main structure of GST Tacotron forked from https://github.com/syang1993/gst-tacotron

A tensorflow implementation of the [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) and [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).

## Audio Samples
 In progress. 
 
## Purpose

A recent trend has been in the development Text-to-Speech (TTS) models in ML with the end goal being to emulate voices and produce mimicked speech that produces words not seen during training. Potential applications of TTS systems include offering more personalized healthcare for those who do not have access to a physical doctor, or using realistic voice simlulation to teach online courses. It should be noted that TTS systems have the potential to infringe upon our identity and authenticity. For example, the recent explosion of Deepfakes have the ability to create "fake news" in which a person of power is artificialy simulated.

## Model Pros and Limitations
 In progress.

## Running the model
Please note: The following instructions are verbatim from @syang1993 in his github repository of GST Tacotron. 

### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better performance, install with GPU support if it's available. This code works with TensorFlow 1.4.

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```

### Training

1. **Download a dataset:**

   The following are supported out of the box:
    * [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) (Public Domain)
    * [Blizzard 2013](https://www.synsig.org/index.php/Blizzard_Challenge_2013) (Creative Commons Attribution Share-Alike)
    * [VCTK]


2. **Preprocess the data**
    
   ```
   python3 preprocess.py --dataset LJSpeech-1.1
   ```

3. **Train a model**

   ```
   python3 train.py
   ```
   
   The above command line will use default hyperparameters, which will train a model with cmudict-based phoneme sequence and 4-head multi-head sytle attention for global style tokens. If you set the `use_gst=False` in the hparams, it will train a model like Google's another paper [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).

   Tunable hyperparameters are found in [hparams.py](hparams.py). You can adjust these at the command line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"` . Hyperparameters should generally be set to the same values at both training and eval time.

4. **Synthesize from a checkpoint**

   ```
   python3 eval.py --checkpoint ~/gst-tacotron-wavenet/logs-tacotron/model.ckpt-185000 --text "hello text" --reference_audio /path/to/ref_audio
   ```

    Replace "185000" with the checkpoint number that you want to use. Then this command line will synthesize a waveform with the content "hello text" and the style of the reference audio. If you don't use the `--reference_audio`, it will generate audio with random style weights, which may generate unintelligible audio sometimes. 

   If you set the `--hparams` flag when training, set the same value here.

## Reference
  -  Keithito's implementation of tacotron: https://github.com/keithito/tacotron
  -  Yuxuan Wang, Daisy Stanton, Yu Zhang, RJ Skerry-Ryan, Eric Battenberg, Joel Shor, Ying Xiao, Fei Ren, Ye Jia, Rif A. Saurous. 2018. [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017)
  - RJ Skerry-Ryan, Eric Battenberg, Ying Xiao, Yuxuan Wang, Daisy Stanton, Joel Shor, Ron J. Weiss, Rob Clark, Rif A. Saurous. 2018. [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).
