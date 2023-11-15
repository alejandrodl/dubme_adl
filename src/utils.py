import soundfile as sf
import whisper
import os
import csv
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import torchaudio
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
import speechmetrics


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


def downsample_and_mono_from_path(audio_path, output_sr=16000):
    """
    Downsample an audio file to an output sample rate and make it mono.

    Args:
        audio_path (str): Path to the audio file.
        output_sr (int): Output sample rate.

    Returns:
        numpy.ndarray: The downsampled audio.
    """
    audio, sr = torchaudio.load(audio_path)
    downsampled_audio = convert_audio(audio, sr, output_sr, 1)
    return downsampled_audio.squeeze().numpy()
        

def get_noise_files(original_audio_path, denoised_audio_path, output_path, model_sr=16000):
    """
    Subtract the denoised audio file from the original audio file to get the noise.

    Args:
        original_audio_path (str): Path to the original audio file.
        denoised_audio_path (str): Path to the denoised audio file.
        output_path (str): Path to save the noise file.
        model_sr (int): Sample rate of the pretrained DNS64 model.
    """
    denoised_audio, _ = sf.read(denoised_audio_path)
    original_audio = downsample_and_mono_from_path(original_audio_path, model_sr)
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


def compute_metrics(metrics_dict, denoised_audio_path, original_audio_path):
    """
    Compute absolute and relative metrics using denoised audio and original audio for relative metrics and only denoised
    audio for absolute metrics. Save the computed metrics to a .csv file containing a table with the name of the metrics
    as columns and the name of the denoised audio files as rows, with all computed metrics values there.

    Args:
        metrics_dict (dict): Dictionary containing the name of the metrics as keys and the corresponding functions as values.
        denoised_audio_path (str): Path to the denoised audio file.
        original_audio_path (str): Path to the original audio file.
    """

    # Set the window length for the metrics in seconds
    window_length = 30

    # Load the denoised audio file
    denoised_audio, sr = sf.read(denoised_audio_path)

    # Compute the absolute metrics
    metrics_absolute = speechmetrics.load('absolute', window_length)
    absolute_metrics = metrics_absolute(denoised_audio, rate=sr)

    # Load the original audio file and convert it to the same sample rate and number of channels as those of the original audio
    original_audio = downsample_and_mono_from_path(original_audio_path, sr)

    # Compute the relative metrics
    metrics_relative = speechmetrics.load(['bsseval', 'sisdr'], window_length)
    relative_metrics = metrics_relative(denoised_audio, original_audio, rate=sr)

    # Combine the absolute and relative metrics into a single dictionary
    file_name = denoised_audio_path.split('/')[-1]
    metrics_dict[file_name] = {**absolute_metrics, **relative_metrics}

    return metrics_dict


def save_metrics(metrics_dict, output_path):
    """
    Create a table from the metrics_dict dictionary output from the compute_metrics function with all the metrics values
    inside and which has the names of the metrics as columns and the names of the files as rows. Save the table to a .csv

    Args:
        metrics_dict (dict): Dictionary containing the name of the metrics as keys and the corresponding functions as values.
        output_path (str): The path to save the metrics to a CSV file.

    Returns:
        pandas.DataFrame: A table with all the metrics values inside and which has the names of the metrics as columns and
        the names of the files as rows.
    """
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(output_path)
