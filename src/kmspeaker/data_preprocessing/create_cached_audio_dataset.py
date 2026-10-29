import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torchaudio.transforms import Resample

from kmspeaker.audio.feature_extractors import AudioFeatureExtractor

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def preprocess_audio_files(audio_data_path: str, metadata_file_str: str, output_file_str: str, model_name: str, cache_dir: str = None):
    """
    Processes audio files listed in a metadata CSV, resampling to 16kHz and extracting hidden states.

    Args:
        audio_data_path (str): Path to the folder containing the .wav audio files.
        metadata_file_str (str): Path to the metadata CSV file.
        output_file_str (str): Path to save the output pickle file.
        model_name (str): Name of the Hugging Face model for AudioFeatureExtractor.
        cache_dir (str, optional): Cache directory for the audio model. Defaults to None.
    """
    audio_data_path = Path(audio_data_path)
    metadata_file = Path(metadata_file_str)
    output_file = Path(output_file_str)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Audio base path: {audio_data_path}")
    logger.info(f"Metadata file: {metadata_file}")
    logger.info(f"Output file: {output_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        extractor = AudioFeatureExtractor(model_name=model_name, cache_dir=cache_dir)
        extractor.to(device)
        extractor.eval()
    except Exception as e:
        logger.error(f"Error initializing AudioFeatureExtractor: {e}")
        sys.exit(1)

    try:
        metadata = pd.read_csv(metadata_file)
        if "audio_name" not in metadata.columns:
            logger.error(f"'audio_name' column not found in metadata file {metadata_file}")
            sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Metadata file not found at {metadata_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading metadata file: {e}")
        sys.exit(1)

    all_hidden_states = {}
    total_clips = len(metadata)

    logger.info("Starting processing of audio files...")
    for index, row in metadata.iterrows():
        clip_name = row["audio_name"]
        audio_file_path = audio_data_path / f"{clip_name}.wav"

        logger.info(f"Processing [{index + 1}/{total_clips}]: {clip_name} (Path: {audio_file_path})")

        if not audio_file_path.exists():
            logger.warning(f"Audio file not found for clip '{clip_name}' at {audio_file_path}. Skipping.")
            continue

        try:
            waveform, sr = sf.read(audio_file_path)

            # Ensure waveform is mono. If stereo, take the first channel.
            if waveform.ndim > 1 and waveform.shape[1] > 1:
                waveform = waveform[:, 0]

            waveform = torch.tensor(waveform, dtype=torch.float32)

            if sr != TARGET_SAMPLE_RATE:
                resampler = Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)
                sr = TARGET_SAMPLE_RATE
                logger.info(f"Resampled {clip_name} to {TARGET_SAMPLE_RATE} Hz.")

            waveform = waveform.to(device)

            # The processor handles any further resampling/padding internally; here we
            # process one file at a time, passed as a batch of 1 for w2v-bert-2.0.
            if model_name == "facebook/w2v-bert-2.0":
                raw_audio_input = [waveform]
            elif model_name == "facebook/hubert-base-ls960":
                raw_audio_input = waveform
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

            with torch.no_grad():
                # (N x T_audio_fps x n_hidden_layers x n_hidden_dim), N=1 for a single file.
                hidden_states_output = extractor(raw_audio_input, sampling_rate=sr)

            all_hidden_states[clip_name] = hidden_states_output.squeeze(0).cpu()

        except Exception as e:
            logger.error(f"Error processing clip '{clip_name}': {e}")

    logger.info(f"Processed {len(all_hidden_states)} clips out of {total_clips}.")

    if not all_hidden_states:
        logger.warning("No clips were processed successfully. Output file will not be saved.")
        return

    try:
        torch.save(all_hidden_states, output_file)
        logger.info(f"Successfully saved all hidden states to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving output file: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Extract hidden states from audio files using AudioFeatureExtractor.")
    parser.add_argument("--audio_data_path", type=str, default="data/dataset/audio", help="Path to the folder containing the .wav audio files.")
    parser.add_argument("--metadata_file", type=str, default="data/dataset_metadata.csv", help="Path to the metadata CSV file.")
    parser.add_argument("--output_file", type=str, default="data/preprocessed_data/dataset_w2vbert_hidden_states.pkl", help="Path to save the output pickle file.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/w2v-bert-2.0",
        help="Name of the Hugging Face model for AudioFeatureExtractor (facebook/w2v-bert-2.0 or facebook/hubert-base-ls960).",
    )
    parser.add_argument("--cache_dir", type=str, default="data/audio_model", help="Cache directory for the audio model.")
    args = parser.parse_args()

    preprocess_audio_files(args.audio_data_path, args.metadata_file, args.output_file, args.model_name, args.cache_dir)
