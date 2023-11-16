import os
import time
from utils import (
    clear_make_directory,
    extract_audio_from_video,
    downsample_and_mono_from_path,
    denoise_audio_file,
    get_noise_files,
    transcribe_audio_file,
    compute_audio_quality_metrics,
    compute_transcription_quality_metrics,
    save_metrics,
)


# Separation models
MODEL = "FAIR"  # 'FAIR' or 'CleanUNet'
print("Using {} model".format(MODEL))

# Set extensions
VIDEO_EXT = ".mp4"
AUDIO_EXT = ".wav"
TEXT_EXT = ".txt"

# Set paths
VIDEO_DIRECTORY = "data/videos"
AUDIO_DIRECTORY = "data/audios"
DENOISED_AUDIO_DIRECTORY = "data/denoised"
NOISE_AUDIO_DIRECTORY = "data/noise"
TRANSCRIPTIONS_DIRECTORY = "data/transcriptions"

# Create directories if they don't exist
clear_make_directory(AUDIO_DIRECTORY, AUDIO_EXT)
clear_make_directory(DENOISED_AUDIO_DIRECTORY, AUDIO_EXT)
clear_make_directory(NOISE_AUDIO_DIRECTORY, AUDIO_EXT)
clear_make_directory(TRANSCRIPTIONS_DIRECTORY, TEXT_EXT)

# STEP 1 - Get video paths and extract audio from each video
video_paths = [
    os.path.join(VIDEO_DIRECTORY, filename)
    for filename in os.listdir(VIDEO_DIRECTORY)
    if filename.endswith(VIDEO_EXT)
]

for video_path in video_paths:
    extract_audio_from_video(video_path, AUDIO_DIRECTORY)

# STEP 2 - Convert all audio files to 16kHz mono
audio_paths = [
    os.path.join(AUDIO_DIRECTORY, filename)
    for filename in os.listdir(AUDIO_DIRECTORY)
    if filename.endswith(AUDIO_EXT)
]

for audio_path in audio_paths:
    downsample_and_mono_from_path(audio_path)

# STEP 3 - Get audio paths, denoise each audio and time it
start_time = time.time()
for audio_path in audio_paths:
    denoise_audio_file(audio_path, DENOISED_AUDIO_DIRECTORY, MODEL)
print("Denoising took {} seconds".format(time.time() - start_time))

# STEP 4 - Get denoised audio paths and subtract them from original audio to get noise file
denoised_audio_paths = [
    os.path.join(DENOISED_AUDIO_DIRECTORY, filename)
    for filename in os.listdir(DENOISED_AUDIO_DIRECTORY)
    if filename.endswith(AUDIO_EXT)
]

for audio_path in audio_paths:
    denoised_audio_path = DENOISED_AUDIO_DIRECTORY + "/" + audio_path.split("/")[-1]
    noise_path = NOISE_AUDIO_DIRECTORY + "/" + denoised_audio_path.split("/")[-1]
    get_noise_files(audio_path, denoised_audio_path, noise_path)

# STEP 5 - Transcribe audio files and denoised audio files and save it to TRANSCRIPTIONS_DIRECTORY
for audio_path in audio_paths:
    transcriptions_path = (
        TRANSCRIPTIONS_DIRECTORY
        + "/"
        + audio_path.split("/")[-1].split(".")[0]
        + ".txt"
    )
    transcribe_audio_file(audio_path, transcriptions_path)

for denoised_audio_path in denoised_audio_paths:
    transcriptions_path = (
        TRANSCRIPTIONS_DIRECTORY
        + "/"
        + denoised_audio_path.split("/")[-1].split(".")[0]
        + "_denoised.txt"
    )
    transcribe_audio_file(denoised_audio_path, transcriptions_path)

# STEP 6 (Optional) - Compute audio quality metrics and save them to a csv file
denoised_audio_paths = [
    os.path.join(DENOISED_AUDIO_DIRECTORY, filename)
    for filename in os.listdir(DENOISED_AUDIO_DIRECTORY)
    if filename.endswith(AUDIO_EXT)
]

audio_metrics_dict: dict = {}
for denoised_audio_path in denoised_audio_paths:
    original_audio_path = AUDIO_DIRECTORY + "/" + denoised_audio_path.split("/")[-1]
    audio_metrics_dict = compute_audio_quality_metrics(
        audio_metrics_dict,
        denoised_audio_path,
        original_audio_path,
        abs_metrics=[],
        rel_metrics=["stoi", "sisdr"],
    )

metrics_path = "data/audio_metrics_" + MODEL + ".csv"
if os.path.exists(metrics_path):
    os.remove(metrics_path)
    save_metrics(audio_metrics_dict, metrics_path)
else:
    save_metrics(audio_metrics_dict, metrics_path)

# STEP 7 (Optional) - Compute transcription quality metrics and save them to a csv file
transcription_metrics_dict: dict = {}
for denoised_audio_path in denoised_audio_paths:
    transcription_gt_path = (
        TRANSCRIPTIONS_DIRECTORY
        + "_gt/"
        + denoised_audio_path.split("/")[-1].split(".")[0]
        + "_gt.txt"
    )
    transcription_path = (
        TRANSCRIPTIONS_DIRECTORY
        + "/"
        + denoised_audio_path.split("/")[-1].split(".")[0]
        + ".txt"
    )
    transcription_denoised_path = transcription_path[:-4] + "_denoised.txt"

    transcription_metrics_dict = compute_transcription_quality_metrics(
        transcription_metrics_dict,
        transcription_gt_path,
        transcription_path,
    )
    transcription_metrics_dict = compute_transcription_quality_metrics(
        transcription_metrics_dict,
        transcription_gt_path,
        transcription_denoised_path,
    )

metrics_path = "data/transcription_metrics_" + MODEL + ".csv"
if os.path.exists(metrics_path):
    os.remove(metrics_path)
    save_metrics(transcription_metrics_dict, metrics_path)
else:
    save_metrics(transcription_metrics_dict, metrics_path)
