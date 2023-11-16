import soundfile as sf
import whisper
import os
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import torchaudio
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
import speechmetrics


def clear_make_directory(directory: str, extension: str):
    """
    Remove all files with a certain extension from a directory.

    Args:
        directory (str): Path to the directory.
        extension (str): Extension of the files to be removed.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                os.remove(os.path.join(directory, filename))


def extract_audio_from_video(video_path: str, audio_directory: str):
    """
    Extract audio from a video file using moviepy.

    Args:
        video_path (str): Path to the video file.
        audio_directory (str): Path to save the extracted audio file.
    """
    video = VideoFileClip(video_path)
    name = video_path.split("/")[-1].split(".")[0] + ".wav"
    output_name = audio_directory + "/" + name
    video.audio.write_audiofile(output_name)


def downsample_and_mono_from_path(audio_path: str, output_sr: int = 16000):
    """
    Downsample an audio file to an output sample rate, make it mono and save it to a .wav file.

    Args:
        audio_path (str): Path to the audio file.
        output_sr (int): Output sample rate.

    Returns:
        numpy.ndarray: The downsampled audio.
    """
    audio, sr = torchaudio.load(audio_path)
    downsampled_audio = convert_audio(audio, sr, output_sr, 1)
    downsampled_audio = downsampled_audio.squeeze().numpy()
    sf.write(audio_path, downsampled_audio, output_sr)


def denoise_audio_file(
    audio_path: str, denoised_audio_directory: str, model_str: str = "CleanUNet"
):
    """
    Denoise an audio file using the pretrained FAIR model (DNS64) or CleanUNet.

    Args:
        audio_path (str): Path to the audio file.
        denoised_audio_directory (str): Path to save the denoised audio file.
    """
    if model_str == "FAIR":
        model = pretrained.dns64()
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, model.sample_rate, model.chin)
        with torch.no_grad():
            denoised = model(wav[None])[0]
        denoised = denoised.squeeze().numpy()
        name = audio_path.split("/")[-1]
        output_name = denoised_audio_directory + "/" + name
        sf.write(output_name, denoised, model.sample_rate)

    elif model_str == "CleanUNet":
        os.chdir(denoised_audio_directory)
        file_name = "../audios/" + audio_path.split("/")[-1]
        command = f"python ../../submodules/CleanUNet/denoise_simple.py -c ../../submodules/CleanUNet/configs/DNS-large-full.json --ckpt_pat ../../submodules/CleanUNet/exp/DNS-large-high/checkpoint/pretrained.pkl {file_name}"
        os.system(command)
        os.chdir("../..")


def get_noise_files(
    original_audio_path: str,
    denoised_audio_path: str,
    output_path: str,
    model_sr: int = 16000,
):
    """
    Subtract the denoised audio file from the original audio file to get the noise.

    Args:
        original_audio_path (str): Path to the original audio file.
        denoised_audio_path (str): Path to the denoised audio file.
        output_path (str): Path to save the noise file.
        model_sr (int): Sample rate of the pretrained DNS64 model.
    """
    denoised_audio, _ = sf.read(denoised_audio_path)
    original_audio, _ = sf.read(original_audio_path)
    noise = original_audio - denoised_audio
    sf.write(output_path, noise, model_sr)


def transcribe_audio_file(audio_path: str, output_path: str):
    """
    Transcribe an audio file in .wav format using whisper and save the text to a .txt document.

    Args:
        audio_path (str): Path to the audio file.
        output_path (str): Path to save the transcribed text file.
    """
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, "w") as f:
        model = whisper.load_model("base.en")
        result = model.transcribe(audio_path)
        f.write(result["text"])


def compute_metrics(
    metrics_dict: dict,
    denoised_audio_path: str,
    original_audio_path: str,
    abs_metrics: list = [],
    rel_metrics: list = ["stoi", "sisdr"],
):
    """
    Compute absolute and relative metrics using denoised audio and original audio for relative
    metrics and only denoised audio for absolute metrics. Save the computed metrics to a .csv
    file containing a table with the name of the metrics as columns and the name of the
    denoised audio files as rows, with all computed metrics values there.

    Args:
        metrics_dict (dict): Dictionary containing the name of the metrics as keys and the
                             corresponding functions as values.
        denoised_audio_path (str): Path to the denoised audio file.
        original_audio_path (str): Path to the original audio file.
    """

    # Set the window length for the metrics in seconds
    window_length = 30

    # Load original and denoised audio files
    original_audio, sr = sf.read(original_audio_path)
    denoised_audio, sr = sf.read(denoised_audio_path)

    # Compute the absolute metrics
    metrics_absolute = speechmetrics.load(abs_metrics, window_length)
    absolute_metrics = metrics_absolute(denoised_audio, rate=sr)

    # Compute the relative metrics
    metrics_relative = speechmetrics.load(rel_metrics, window_length)
    relative_metrics = metrics_relative(denoised_audio, original_audio, rate=sr)

    # Combine the absolute and relative metrics into a single dictionary
    file_name = denoised_audio_path.split("/")[-1]
    metrics_dict[file_name] = {**absolute_metrics, **relative_metrics}

    return metrics_dict


def save_metrics(metrics_dict: dict, output_path: str):
    """
    Create a table from the metrics_dict dictionary output from the compute_metrics function
    with all the metrics values inside and which has the names of the metrics as columns and
    the names of the files as rows. Save the table to a .csv

    Args:
        metrics_dict (dict): Dictionary containing the name of the metrics as keys and the
                             corresponding functions as values.
        output_path (str): The path to save the metrics to a CSV file.

    Returns:
        pandas.DataFrame: A table with all the metrics values inside and which has the names
        of the metrics as columns and the names of the files as rows.
    """
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(output_path)
