import os
import time
import shutil
import argparse
from utils import (
    clear_make_directory,
    extract_audio_from_video,
    downsample_and_mono_from_path,
    denoise_audio_file,
    get_noise_files,
)


def main(video_path):
    # Separation models
    MODEL = "FAIR"  # 'FAIR' or 'CleanUNet'
    print("Using {} model".format(MODEL))

    # Set extensions
    VIDEO_EXT = ".mp4"
    AUDIO_EXT = ".wav"

    # Set paths
    MAIN_PATH = 'output_of_separation_program'
    AUDIO_DIRECTORY = MAIN_PATH + "/audio_file"
    DENOISED_AUDIO_DIRECTORY = MAIN_PATH + "/denoised_file"
    NOISE_AUDIO_DIRECTORY = MAIN_PATH + "/noise_file"

    # Create directories if they don't exist
    if not os.path.exists(MAIN_PATH):
        os.makedirs(MAIN_PATH)
    clear_make_directory(AUDIO_DIRECTORY, AUDIO_EXT)
    clear_make_directory(DENOISED_AUDIO_DIRECTORY, AUDIO_EXT)
    clear_make_directory(NOISE_AUDIO_DIRECTORY, AUDIO_EXT)

    # STEP 1 - Extract audio from video
    extract_audio_from_video(video_path, AUDIO_DIRECTORY)

    # STEP 2 - Convert audio file to 16kHz mono
    audio_path = os.path.join(
        AUDIO_DIRECTORY, os.path.basename(video_path).replace(VIDEO_EXT, AUDIO_EXT)
    )
    downsample_and_mono_from_path(audio_path)

    # STEP 3 - Denoise audio file
    start_time = time.time()
    denoise_audio_file(audio_path, DENOISED_AUDIO_DIRECTORY, MODEL)
    print("Denoising took {} seconds".format(time.time() - start_time))

    # STEP 4 - Calculate noise file
    denoised_audio_path = os.path.join(
        DENOISED_AUDIO_DIRECTORY, os.path.basename(audio_path)
    )
    noise_path = os.path.join(NOISE_AUDIO_DIRECTORY, os.path.basename(audio_path))
    get_noise_files(audio_path, denoised_audio_path, noise_path)
    shutil.rmtree(AUDIO_DIRECTORY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate noise from a video's audio")
    parser.add_argument("-vp", "--video-path", type=str, help="path to video file")
    args = parser.parse_args()
    main(args.video_path)
