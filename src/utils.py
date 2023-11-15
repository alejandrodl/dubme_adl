import soundfile as sf
import whisper
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

import torchaudio
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio


def extract_audio_from_video(video_path, audio_directory):
    """
    Extract audio from a video file using moviepy.

    Args:
        video_path (str): Path to the video file.
        audio_directory (str): Path to save the extracted audio file.
    """
    video = VideoFileClip(video_path)
    name = video_path.split('/')[-1].split('.')[0] + '.wav'
    output_name = audio_directory + '/' + name
    video.audio.write_audiofile(output_name)


def denoise_audio_file(audio_path, denoised_audio_directory):
    """
    Denoise an audio file using the pretrained DNS64 model.

    Args:
        audio_path (str): Path to the audio file.
        denoised_audio_directory (str): Path to save the denoised audio file.
    """
    model = pretrained.dns64()
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
    denoised = denoised.squeeze().numpy()
    name = audio_path.split('/')[-1]
    output_name = denoised_audio_directory + '/' + name
    sf.write(output_name, denoised, model.sample_rate)
        

def get_noise_files(original_audio_path, denoised_audio_path, output_path, model_sr=16000, model_chin=1):
    """
    Subtract the denoised audio file from the original audio file to get the noise.

    Args:
        original_audio_path (str): Path to the original audio file.
        denoised_audio_path (str): Path to the denoised audio file.
        output_path (str): Path to save the noise file.
        model_sr (int): Sample rate of the pretrained DNS64 model.
        model_chin (int): Number of input channels of the pretrained DNS64 model.
    """
    denoised_audio, _ = sf.read(denoised_audio_path)
    original_audio, sr = torchaudio.load(original_audio_path)
    original_audio = convert_audio(original_audio, sr, model_sr, model_chin)
    original_audio = original_audio.squeeze().numpy()
    noise = original_audio - denoised_audio
    sf.write(output_path, noise, model_sr)


def transcribe_audio_file(audio_path, output_path):
    """
    Transcribe an audio file in .wav format using whisper and save the text to a .txt document.

    Args:
        audio_path (str): Path to the audio file.
        output_path (str): Path to save the transcribed text file.
    """
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, 'w') as f:
        model = whisper.load_model("base.en")
        result = model.transcribe(audio_path)
        f.write(result["text"])
