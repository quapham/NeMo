# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Used in inference and evaluation scripts to obtain metrics such as ASR_WER and UTMOSV2 scores.
"""
import argparse
import json
import multiprocessing
import os
import pprint
import re
import tempfile
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.tts.metrics.eou_classifier import EoUClassification, EoUClassifier, EoUType
from nemo.collections.tts.metrics.frechet_codec_distance import FrechetCodecDistance
from nemo.collections.tts.metrics.prosody import compute_prosody_distances
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    JapaneseTextProcessor,
    NemoTranscriber,
    NemoTranscriberWithPrompt,
    WhisperTranscriber,
    get_text_processor,
)
from nemo.utils import logging

# Optional import for UTMOSv2 (audio quality metric)
try:
    from nemo.collections.tts.modules.utmosv2 import UTMOSv2Calculator

    UTMOSV2_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    UTMOSV2_AVAILABLE = False
    logging.warning(
        f"UTMOSv2Calculator not available: {e}. "
        "UTMOSv2 metrics will be disabled. Install required dependencies to enable."
        "To install utmosv2 run `pip install git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1`."
    )


KATAKANA_METRICS_TO_SAVE = [
    'katakana_cer',
    'gt_katakana',
    'pred_katakana',
]

PROSODY_DISTANCE_KEYS = [
    'pitch_distance',
    'intensity_distance',
    'speech_rate_distance',
]

# Regexes mirrored from the IPA preprocessing script that creates
# custom["text_without_annotation"]. This is used only for text inputs
# during metric computation when requested.
_WS_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?؟،؛])")
_TATWEEL_RE = re.compile("\u0640+")
_ANNOTATION_OR_MARKER_RE = re.compile(
    r"""
      \[[^\[\]\n]{1,512}\]          # square annotation: [breath], [نقر]
    | </?[^<>\n]{1,512}>            # XML/style/language tags
    | \{/?[^{}\n]{1,512}\}          # curly control/pronunciation tags
    | [-–—]{2,}                     # multi-dash cutoff: --, ---, ——
    | (?<=\S)[-–—](?=\s|$)          # trailing single dash after a token: word-
    | (?:^|(?<=\s))[-–—](?=\s|$)    # standalone dash
    | \.{3,}                        # ASCII ellipsis
    | …+                            # Unicode ellipsis
    | \*+                            # emphasis marker: *word*
    """,
    re.VERBOSE,
)


def strip_text_annotations_from_text(text: str) -> str:
    """Return orthographic text with annotation/control tokens removed."""
    text = _ANNOTATION_OR_MARKER_RE.sub(" ", str(text))
    text = _TATWEEL_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text.strip()


FILEWISE_METRICS_TO_SAVE = [
    'cer',
    'cer_pred_gt_audio',
    'wer',
    'wer_pred_gt_audio',
    'pred_context_ssim',
    'pred_gt_esim',
    'pred_gt_ems',
    *PROSODY_DISTANCE_KEYS,
    'pred_text',
    'gt_audio_text',
    'gt_text',
    'predicted_phoneme_text',
    'predicted_phoneme_tokens',
    'predicted_phoneme_token_labels',
    'gt_audio_filepath',
    'pred_audio_filepath',
    'context_audio_filepath',
    'utmosv2',
    'eou_type',
    'eou_trailing_duration',
    'eou_trail_rms_ratio',
]


def load_evalset_config(config_path: Optional[str] = None, dataset_base_path: Optional[Path] = None) -> dict:
    """Load dataset meta info from JSON config file."""
    if config_path is None or not os.path.exists(config_path):
        raise ValueError("No dataset_json_path provided, please provide a valid path to the evalset config file.")

    logging.info(f"Loading evalset config from {config_path}")
    with open(config_path, 'r') as f:
        dataset_meta_info = json.load(f)

    # Validate that all evaluation datasets exist
    for dataset_name, info in dataset_meta_info.items():
        manifest_path = Path(info["manifest_path"])
        audio_dir = Path(info["audio_dir"])

        asr_model_path = None
        asr_model = info.get("asr_model")
        if asr_model:
            asr_model_name = asr_model["name"]
            if asr_model_name.endswith(".nemo"):
                asr_model_path = Path(asr_model_name)

        if dataset_base_path:
            # Replace relative paths with absolute paths where appropriate
            if not manifest_path.is_absolute():
                manifest_path = dataset_base_path / manifest_path
                info["manifest_path"] = str(manifest_path)

            if not audio_dir.is_absolute():
                audio_dir = dataset_base_path / audio_dir
                info["audio_dir"] = str(audio_dir)

            if asr_model_path and not asr_model_path.is_absolute():
                asr_model_path = dataset_base_path / asr_model_path
                info["asr_model"]["name"] = str(asr_model_path)

        if not manifest_path.exists():
            raise ValueError(f"Manifest does not exist for dataset {dataset_name}: {manifest_path}")

        if not audio_dir.exists():
            raise ValueError(f"Audio directory does not exist for dataset {dataset_name}: {audio_dir}")

        if asr_model_path and not asr_model_path.exists():
            raise ValueError(f"ASR model file does not exist for dataset {dataset_name}: {asr_model_path}")

    return dataset_meta_info


def _resolve_path(audio_dir, path):
    """Resolve a relative path against audio_dir if both are provided."""
    if audio_dir is not None and path is not None:
        return os.path.join(audio_dir, path)
    return path


def find_generated_files(audio_dir, prefix, extension):
    file_list = []
    for f in os.listdir(audio_dir):
        if prefix in f and f.endswith(extension):
            audio_number = int(f.split("_")[-1].split(extension)[0])
            file_list.append((audio_number, os.path.join(audio_dir, f)))
    file_list.sort()
    file_list = [t[1] for t in file_list]
    return file_list


def find_generated_audio_files(audio_dir):
    return find_generated_files(audio_dir=audio_dir, prefix="predicted_audio", extension=".wav")


def find_generated_codec_files(audio_dir):
    return find_generated_files(audio_dir=audio_dir, prefix="predicted_codes", extension=".pt")


def get_wav_file_duration(audio_path: str) -> float:
    """
    Get the duration of an WAV file in seconds.
    """
    # get extension of the file
    extension = os.path.splitext(audio_path)[1]
    if extension.lower() != ".wav":
        raise ValueError(f"Audio path {audio_path} is not a WAV file")
    info = sf.info(audio_path)
    seconds = info.frames / info.samplerate
    return seconds


def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            records.append(json.loads(line))
    return records


def pad_audio_to_min_length(audio_np: np.ndarray, sampling_rate: int, min_seconds: float) -> np.ndarray:
    """
    Pad audio to make it at least `min_seconds` long by adding silence at the end if needed.
    """
    if audio_np.ndim != 1:
        raise ValueError("Audio array must be 1D")

    n_samples = len(audio_np)
    min_samples = round(min_seconds * sampling_rate)

    if n_samples < min_samples:
        logging.info(f"Padding audio from {n_samples/sampling_rate} seconds to {min_samples/sampling_rate} seconds")
        padding_needed = min_samples - n_samples
        audio_np = np.pad(audio_np, (0, padding_needed), mode='constant', constant_values=0)
    return audio_np


def extract_embedding(model, extractor, audio_path, device, sv_model_type):
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    # pad to 0.5 seconds as the extractor may not be able to handle very short signals
    speech_array = pad_audio_to_min_length(speech_array, int(sampling_rate), min_seconds=0.5)
    if sv_model_type == "wavlm":
        inputs = extractor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_values.to(device)
        with torch.inference_mode():
            embeddings = model(inputs).embeddings
    else:  # Titanet
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            # the embedding model doesn't accept NumPy arrays, so we write to a temporary file
            sf.write(temp_file.name, speech_array, samplerate=16000)
            with torch.inference_mode():
                embeddings = model.get_embedding(temp_file.name).squeeze()

    return embeddings.squeeze()


def compute_utmosv2_scores(audio_dir, device):
    if not UTMOSV2_AVAILABLE:
        logging.warning("UTMOSv2Calculator not available. Skipping UTMOSv2 score computation.")
        return {}

    logging.info(f"\nComputing UTMOSv2 scores for files in {audio_dir}...")
    start_time = time.time()
    utmosv2_calculator = UTMOSv2Calculator(device=device)
    utmosv2_scores = utmosv2_calculator.process_directory(audio_dir)
    # convert to to a dictionary indexed by file path
    utmosv2_scores_dict = {os.path.normpath(item['file_path']): item['predicted_mos'] for item in utmosv2_scores}
    end_time = time.time()
    logging.info(f"UTMOSv2 scores computed for {len(utmosv2_scores)} files in {end_time - start_time:.2f} seconds\n")
    return utmosv2_scores_dict


def load_evaluation_models(
    sv_model_type="titanet",
    asr_model_name="stt_en_conformer_transducer_large",
    asr_model_type="nemo",
    device="cuda",
    with_prosody_metrics=False,
    prosody_model_size="small",
):
    """Load the ASR and speaker-verification models used for evaluation.

    Args:
        sv_model_type: Speaker-verification model type. Supported values are
            ``"wavlm"`` and ``"titanet"``.
        asr_model_name: Name or path of the ASR model to load.
        asr_model_type: ASR model implementation. Supported values are
            ``"nemo"``, ``"nemo_with_prompt"``, and ``"whisper"``.
        device: Device on which the evaluation models are loaded.
        with_prosody_metrics: Whether to compute ESIM/EMS plus pitch,
            intensity, and speech-rate distance metrics.
        prosody_model_size: Size of the emotion encoder. Supported values are ``"small"`` or ``"large"``.

    Returns:
        Dictionary containing:

            - ``asr_model``: Loaded ASR transcriber.
            - ``whisper_model``: Reserved Whisper model entry, currently ``None``.
            - ``whisper_processor``: Reserved Whisper processor entry, currently
            ``None``.
            - ``feature_extractor``: WavLM feature extractor when
            ``sv_model_type="wavlm"``; otherwise ``None``.
            - ``sv_model``: Primary speaker-verification model.
            - ``sv_model_alternate``: Alternate ``titanet_small``
            speaker-verification model.
            - ``emotion_model``: Emotion encoder when ``with_prosody_metrics=True``;
            otherwise ``None``. It is also ``None`` if loading fails.

    Raises:
        ValueError: If ``asr_model_type`` is unsupported.
    """
    models = {
        'asr_model': None,
        'whisper_model': None,
        'whisper_processor': None,
        'feature_extractor': None,
        'emotion_model': None,
    }

    models['asr_model'] = load_asr_transcriber(asr_model_name, asr_model_type, device)
    models.update(load_speaker_verification_models(sv_model_type, device))

    if with_prosody_metrics:
        logging.info("Loading emotion encoder for ESIM/EMS prosody metrics...")
        try:
            from nemo.collections.tts.metrics.emotion_encoder import EmpathicInsightVoice

            models['emotion_model'] = EmpathicInsightVoice.from_pretrained(
                size=prosody_model_size,
                device=device,
                mlp_device=device,
                cache_classifiers=True,
                load_all_classifiers=False,
                top_k_emotions=1,
            ).eval()
        except Exception as e:
            logging.warning(f"Emotion encoder could not be loaded: {e}. ESIM/EMS metrics will be set to NaN.")

    return models


def load_speaker_verification_models(sv_model_type="titanet", device="cuda") -> dict:
    """Load the speaker-verification models: ``feature_extractor`` (WavLM only), ``sv_model`` and the alternate
    ``titanet_small`` (``sv_model_alternate``)."""
    models = {'feature_extractor': None}
    if sv_model_type == "wavlm":
        models['feature_extractor'] = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        models['sv_model'] = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device).eval()
    else:
        models['sv_model'] = (
            nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large').to(device).eval()
        )

    logging.info("Loading `titanet_small` model...")
    with logging.temp_verbosity(logging.ERROR):
        models['sv_model_alternate'] = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name='titanet_small'
        )
    models['sv_model_alternate'] = models['sv_model_alternate'].to(device).eval()
    return models


def load_asr_transcriber(asr_model_name, asr_model_type="nemo", device="cuda"):
    """Load the ASR transcriber of ``asr_model_type`` ("nemo", "nemo_with_prompt" or "whisper")."""
    if asr_model_type == "nemo":
        return NemoTranscriber(model_name=asr_model_name, device=device)
    if asr_model_type == "nemo_with_prompt":
        return NemoTranscriberWithPrompt(model_name=asr_model_name, device=device)
    if asr_model_type == "whisper":
        return WhisperTranscriber(model_name=asr_model_name, device=device)
    raise ValueError(f"Unknown ASR model type {asr_model_name}")


def compute_emotion_pair_metrics(emotion_model, gt_audio_path, pred_audio_path):
    """Compute ground-truth to predicted emotion similarity and top-emotion match."""
    if emotion_model is None or gt_audio_path is None or pred_audio_path is None:
        return float('NaN'), float('NaN')

    try:
        result = emotion_model.compare_emotion_pair(
            audio_path_a=gt_audio_path,
            audio_path_b=pred_audio_path,
        )
        return float(result["emotion_similarity"]), float(result["top_emotion_match"])
    except Exception as e:
        logging.warning(f"Could not compute ESIM/EMS for {gt_audio_path} and {pred_audio_path}: {e}")
        return float('NaN'), float('NaN')


def _empty_prosody_distance_metrics():
    return {key: float('NaN') for key in PROSODY_DISTANCE_KEYS}


def compute_acoustic_prosody_metrics(
    gt_audio_path,
    pred_audio_path,
    text,
):
    """Compute reference-based pitch, intensity, and speech-rate distances."""
    if gt_audio_path is None or pred_audio_path is None:
        return _empty_prosody_distance_metrics()

    try:
        metrics = compute_prosody_distances(
            gt_audio_path=gt_audio_path,
            pred_audio_path=pred_audio_path,
            text=text,
        ).to_dict()
        return {key: metrics[key] for key in PROSODY_DISTANCE_KEYS}
    except Exception as e:
        logging.warning(f"Could not compute acoustic prosody distances for {gt_audio_path} and {pred_audio_path}: {e}")
        return _empty_prosody_distance_metrics()


def classify_eou_batched(
    eou_classifier: EoUClassifier, items: list[tuple[Union[str, np.ndarray], str]], batch_size: int = 32
) -> list[EoUClassification]:
    """Run EoU classification in batches.

    Args:
        eou_classifier: EoUClassifier instance.
        items: List of (audio, text) pairs. Audio can be a file path or numpy array.
        batch_size: Batch size.
    """
    logging.info("\nRunning End-of-Utterance (EoU) classification...")
    start_time = time.time()
    results = []
    for start in range(0, len(items), batch_size):
        results.extend(eou_classifier.classify_batch(items[start : start + batch_size]))
    elapsed = time.time() - start_time
    logging.info(f"EoU classification for {len(results)} files took {elapsed:.2f} seconds\n")
    return results


def evaluate_dir(
    manifest_path,
    audio_dir,
    generated_audio_dir,
    language="en",
    sv_model_type="titanet",
    asr_model_name="stt_en_conformer_transducer_large",
    asr_model_type="nemo",
    with_utmosv2=True,
    strip_text_annotations_for_metrics=False,
    with_prosody_metrics=False,
    prosody_model_size="small",
    asr_batch_size=32,
    eou_batch_size=32,
    device="cuda",
    eou_model_name=None,
):
    """Compute per-file evaluation metrics for a directory of generated audio.

    Evaluates individual files (ASR WER/CER, speaker similarity, UTMOSv2) but
    does not compute global/aggregate metrics (cumulative WER/CER, FCD). Use
    compute_global_metrics() to aggregate results from one or more directories.

    External callers (e.g. NeMo Skills) can run evaluate_dir() in parallel across
    multiple directories (on different workers/GPUs), concatenate the resulting
    filewise_metrics lists, and call compute_global_metrics() once at the end to
    accumulate the global metrics.

    Returns:
        filewise_metrics: List of per-file metric dictionaries.
    """
    logging.info(f"Evaluating generated audio in {generated_audio_dir}...")

    # 1. Collect files to evaluate
    audio_file_lists = find_generated_audio_files(generated_audio_dir)
    records = read_manifest(manifest_path)
    assert len(audio_file_lists) == len(records)
    codes_file_lists = find_generated_codec_files(generated_audio_dir)
    has_codes = len(codes_file_lists) == len(records)
    # Resolve ground-truth and context audio paths for all records
    gt_audio_paths = [_resolve_path(audio_dir, r.get('audio_filepath')) for r in records]
    context_audio_paths = [_resolve_path(audio_dir, r.get('context_audio_filepath')) for r in records]

    # 2. Load models
    models = load_evaluation_models(
        sv_model_type=sv_model_type,
        asr_model_name=asr_model_name,
        asr_model_type=asr_model_type,
        device=device,
        with_prosody_metrics=with_prosody_metrics,
        prosody_model_size=prosody_model_size,
    )

    asr_model = models['asr_model']
    feature_extractor = models['feature_extractor']
    speaker_verification_model = models['sv_model']
    speaker_verification_model_alternate = models['sv_model_alternate']
    emotion_model = models['emotion_model']

    # 3. EoU classifier (support for English only)
    if language == "en":
        eou_kwargs = {"device": device}
        if eou_model_name is not None:
            eou_kwargs["model_name"] = eou_model_name
        eou_classifier = EoUClassifier(**eou_kwargs)
    else:
        eou_classifier = None

    # 4. Compute UTMOSv2 scores
    utmosv2_scores = None
    if with_utmosv2:
        if not UTMOSV2_AVAILABLE:
            logging.warning(
                "UTMOSv2 was requested (with_utmosv2=True) but the UTMOSv2 library is not available. "
                "UTMOSv2 scores will be set to NaN for all files."
            )
        utmosv2_scores = compute_utmosv2_scores(generated_audio_dir, device)

    # 5. ASR transcription in batches
    logging.info(f"Doing batched ASR transcription with batch size {asr_batch_size}...")

    # Transcribe predicted audios
    text_processor = get_text_processor(language)
    pred_texts = asr_model.transcribe(audio_paths=audio_file_lists, language=language, batch_size=asr_batch_size)
    if strip_text_annotations_for_metrics:
        pred_texts = [strip_text_annotations_from_text(text) for text in pred_texts]
    pred_texts = [text_processor.process_text_for_wer(text) for text in pred_texts]
    # Transcribe ground truth audios
    if len(gt_audio_paths) > 0:
        gt_audio_texts = asr_model.transcribe(audio_paths=gt_audio_paths, language=language, batch_size=asr_batch_size)
        if strip_text_annotations_for_metrics:
            gt_audio_texts = [strip_text_annotations_from_text(text) for text in gt_audio_texts]
        gt_audio_texts = [text_processor.process_text_for_wer(text) for text in gt_audio_texts]
    else:
        gt_audio_texts = [None] * len(records)

    # 6. Pre-compute ground-truth texts for all records
    gt_texts_processed = []
    for record in records:
        if "original_text" in record:
            text_field = 'original_text'
        elif 'normalized_text' in record:
            text_field = 'normalized_text'
        else:
            text_field = 'text'
        text = record[text_field]
        if strip_text_annotations_for_metrics:
            text = strip_text_annotations_from_text(text)
        processed_text = text_processor.process_text_for_wer(text)
        gt_texts_processed.append(processed_text)

    # 7. Batched EoU classification
    eou_results = None
    if eou_classifier is not None:
        eou_items = list(zip(audio_file_lists, gt_texts_processed))
        eou_results = classify_eou_batched(eou_classifier, eou_items, batch_size=eou_batch_size)

    # 8. Compute metrics for each utterance (sequential)
    filewise_metrics = []
    total_generated_audio_seconds = 0.0
    for ridx, record in enumerate(records):
        gt_audio_filepath = gt_audio_paths[ridx]
        context_audio_filepath = context_audio_paths[ridx]

        pred_audio_filepath = audio_file_lists[ridx]
        pred_text = pred_texts[ridx]
        gt_audio_text = gt_audio_texts[ridx]

        if with_utmosv2 and UTMOSV2_AVAILABLE:
            utmosv2_score = utmosv2_scores[os.path.normpath(pred_audio_filepath)]
        else:
            utmosv2_score = float('nan')

        gt_text = gt_texts_processed[ridx]

        detailed_cer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=True)
        detailed_wer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=False)
        cer_pred_gt_audio = (
            word_error_rate_detail(hypotheses=[pred_text], references=[gt_audio_text], use_cer=True)[0]
            if gt_audio_text is not None
            else float('NaN')
        )
        wer_pred_gt_audio = (
            word_error_rate_detail(hypotheses=[pred_text], references=[gt_audio_text], use_cer=False)[0]
            if gt_audio_text is not None
            else float('NaN')
        )

        # Japanese: additional reading-based CER on Katakana (pyopenjtalk g2p), robust to
        # kanji/kana spelling differences between reference and ASR hypothesis.
        gt_katakana = pred_katakana = None
        katakana_cer = None
        if isinstance(text_processor, JapaneseTextProcessor):
            gt_katakana = text_processor.text_to_katakana(gt_text)
            pred_katakana = text_processor.text_to_katakana(pred_text)
            katakana_cer = word_error_rate_detail(hypotheses=[pred_katakana], references=[gt_katakana], use_cer=True)[
                0
            ]

        pred_gt_esim = float('NaN')
        pred_gt_ems = float('NaN')
        prosody_distance_metrics = _empty_prosody_distance_metrics()
        if with_prosody_metrics:
            pred_gt_esim, pred_gt_ems = compute_emotion_pair_metrics(
                emotion_model,
                gt_audio_filepath,
                pred_audio_filepath,
            )
            prosody_distance_metrics = compute_acoustic_prosody_metrics(
                gt_audio_path=gt_audio_filepath,
                pred_audio_path=pred_audio_filepath,
                text=gt_text,
            )
        logging.info(f"{ridx} GT Text: {gt_text}")
        logging.info(f"{ridx} Pr Text: {pred_text}")
        # Format cer and wer to 2 decimal places
        logging.info(f"CER: {detailed_cer[0]:.4f} | WER: {detailed_wer[0]:.4f}")

        with torch.inference_mode():
            extract_embedding_fn = partial(
                extract_embedding,
                model=speaker_verification_model,
                extractor=feature_extractor,
                device=device,
                sv_model_type=sv_model_type,
            )
            extract_embedding_fn_alternate = partial(
                extract_embedding,
                model=speaker_verification_model_alternate,
                extractor=feature_extractor,
                device=device,
                sv_model_type="titanet",  # alternate is always titanet
            )

            # Initialize SSIMs with a default since the context or ground truth audio
            # may be unavailable.
            pred_context_ssim = float('NaN')
            gt_context_ssim = float('NaN')
            pred_context_ssim_alternate = float('NaN')
            gt_context_ssim_alternate = float('NaN')
            pred_gt_ssim = float('NaN')
            pred_gt_ssim_alternate = float('NaN')

            if gt_audio_filepath is not None:
                # Ground truth vs. predicted
                gt_speaker_embedding = extract_embedding_fn(audio_path=gt_audio_filepath)
                pred_speaker_embedding = extract_embedding_fn(audio_path=pred_audio_filepath)
                pred_gt_ssim = torch.nn.functional.cosine_similarity(
                    gt_speaker_embedding, pred_speaker_embedding, dim=0
                ).item()

                # Ground truth vs. predicted (alternate model)
                gt_speaker_embedding_alternate = extract_embedding_fn_alternate(audio_path=gt_audio_filepath)
                pred_speaker_embedding_alternate = extract_embedding_fn_alternate(audio_path=pred_audio_filepath)
                pred_gt_ssim_alternate = torch.nn.functional.cosine_similarity(
                    gt_speaker_embedding_alternate, pred_speaker_embedding_alternate, dim=0
                ).item()

            if context_audio_filepath is not None:
                context_speaker_embedding = extract_embedding_fn(audio_path=context_audio_filepath)
                context_speaker_embedding_alternate = extract_embedding_fn_alternate(audio_path=context_audio_filepath)

                # Predicted vs. context
                pred_context_ssim = torch.nn.functional.cosine_similarity(
                    pred_speaker_embedding, context_speaker_embedding, dim=0
                ).item()
                # Ground truth vs. context
                if gt_audio_filepath is not None:
                    gt_context_ssim = torch.nn.functional.cosine_similarity(
                        gt_speaker_embedding, context_speaker_embedding, dim=0
                    ).item()

                # Predicted vs. context (alternate model)
                pred_context_ssim_alternate = torch.nn.functional.cosine_similarity(
                    pred_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0
                ).item()
                # Ground truth vs. context (alternate model)
                if gt_audio_filepath is not None:
                    gt_context_ssim_alternate = torch.nn.functional.cosine_similarity(
                        gt_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0
                    ).item()
            file_duration = get_wav_file_duration(pred_audio_filepath)
            total_generated_audio_seconds += file_duration

        if eou_results is not None:
            eou_result = eou_results[ridx]
            if not eou_result.eou_type == EoUType.GOOD:
                logging.warning(
                    f"EoU classification: {eou_result.eou_type.value.upper()} for {pred_audio_filepath} (text: {gt_text})"
                )
            eou_type = eou_result.eou_type.value
            eou_trailing = eou_result.trailing_duration
            eou_rms_ratio = eou_result.trail_rms_ratio
        else:
            eou_type = None
            eou_trailing = float('nan')
            eou_rms_ratio = float('nan')

        metric_row = {
            'gt_text': gt_text,
            'pred_text': pred_text,
            'gt_audio_text': gt_audio_text,
            'predicted_phoneme_text': record.get('predicted_phoneme_text', ''),
            'predicted_phoneme_tokens': record.get('predicted_phoneme_tokens', []),
            'predicted_phoneme_token_labels': record.get('predicted_phoneme_token_labels', []),
            'detailed_cer': detailed_cer,
            'detailed_wer': detailed_wer,
            'cer': detailed_cer[0],
            'cer_pred_gt_audio': cer_pred_gt_audio,
            'wer': detailed_wer[0],
            'wer_pred_gt_audio': wer_pred_gt_audio,
            'katakana_cer': katakana_cer,
            'gt_katakana': gt_katakana,
            'pred_katakana': pred_katakana,
            'pred_gt_ssim': pred_gt_ssim,
            'pred_context_ssim': pred_context_ssim,
            'gt_context_ssim': gt_context_ssim,
            'pred_gt_ssim_alternate': pred_gt_ssim_alternate,
            'pred_context_ssim_alternate': pred_context_ssim_alternate,
            'gt_context_ssim_alternate': gt_context_ssim_alternate,
            'gt_audio_filepath': gt_audio_filepath,
            'pred_audio_filepath': pred_audio_filepath,
            'context_audio_filepath': context_audio_filepath,
            'utmosv2': utmosv2_score,
            'eou_type': eou_type,
            'eou_trailing_duration': eou_trailing,
            'eou_trail_rms_ratio': eou_rms_ratio,
            'total_gen_audio_seconds': file_duration,
            'predicted_codes_path': codes_file_lists[ridx] if has_codes else None,
        }
        if with_prosody_metrics:
            metric_row['pred_gt_esim'] = pred_gt_esim
            metric_row['pred_gt_ems'] = pred_gt_ems
            metric_row.update(prosody_distance_metrics)
        filewise_metrics.append(metric_row)

    return filewise_metrics


def evaluate(
    manifest_path,
    audio_dir,
    generated_audio_dir,
    language="en",
    sv_model_type="titanet",
    asr_model_name="stt_en_conformer_transducer_large",
    asr_model_type="nemo",
    with_utmosv2=True,
    strip_text_annotations_for_metrics=False,
    with_fcd=True,
    codec_model_path=None,
    with_prosody_metrics=False,
    prosody_model_size="small",
    asr_batch_size=32,
    eou_batch_size=32,
    device="cuda",
    eou_model_name=None,
):
    """Evaluate generated audio, computing both per-file and global metrics.

    Convenience wrapper that calls evaluate_dir() for per-file scoring and
    compute_global_metrics() for aggregation of metrics.

    External callers (e.g. NeMo Skills) can run evaluate_dir() in parallel across
    multiple directories (on different workers/GPUs), concatenate the resulting
    filewise_metrics lists, and call compute_global_metrics() once at the end to
    aggregate the global metrics.

    Note the FCD computation is deferred to compute_global_metrics() because the
    metric implementation is stateful and therefore cannot be easily merged across
    subsets (directories).

    Returns:
        Tuple of (avg_metrics dict, filewise_metrics list).
    """
    start_time = time.time()

    if with_fcd and codec_model_path is None:
        raise ValueError("codec_model_path is required when with_fcd is True")

    filewise_metrics = evaluate_dir(
        manifest_path=manifest_path,
        audio_dir=audio_dir,
        generated_audio_dir=generated_audio_dir,
        language=language,
        sv_model_type=sv_model_type,
        asr_model_name=asr_model_name,
        asr_model_type=asr_model_type,
        with_utmosv2=with_utmosv2,
        strip_text_annotations_for_metrics=strip_text_annotations_for_metrics,
        with_prosody_metrics=with_prosody_metrics,
        prosody_model_size=prosody_model_size,
        asr_batch_size=asr_batch_size,
        eou_batch_size=eou_batch_size,
        device=device,
        eou_model_name=eou_model_name,
    )

    gt_audio_paths = None
    predicted_codes_paths = None
    if with_fcd:
        gt_audio_paths = [m['gt_audio_filepath'] for m in filewise_metrics]
        predicted_codes_paths = [m['predicted_codes_path'] for m in filewise_metrics]

    avg_metrics = compute_global_metrics(
        filewise_metrics=filewise_metrics,
        gt_audio_paths=gt_audio_paths,
        predicted_codes_paths=predicted_codes_paths,
        codec_model_path=codec_model_path,
        device=device,
    )

    elapsed = time.time() - start_time
    logging.info(f"evaluate() completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    filewise_metrics_to_save = list(FILEWISE_METRICS_TO_SAVE)
    if (language or "").replace("_", "-").lower().split("-")[0] == "ja":
        filewise_metrics_to_save[2:2] = KATAKANA_METRICS_TO_SAVE

    filtered_filewise = [{k: m[k] for k in filewise_metrics_to_save if k in m} for m in filewise_metrics]
    return avg_metrics, filtered_filewise


# ---------------------------------------------------------------------------
# Long-form chunked inference: per-chunk ASR and join pitch
# ---------------------------------------------------------------------------

LONGFORM_CHUNK_DIR = "longform_chunks"
LONGFORM_FILEWISE_METRICS_TO_SAVE = ['pred_context_ssim_chunks', 'pred_context_ssim_chunks_alternate', 'num_chunks']
LONGFORM_CHUNK_METRICS_FILE = "longform_chunk_metrics.json"
_PYIN_SR, _PYIN_HOP = 16000, 320
_JOIN_END_SEC, _JOIN_START_SEC = 3.5, 2.5  # last 3.5 s of chunk N vs first 2.5 s of chunk N+1
_MIN_F0_FRAMES = 10  # 0.2 s of voiced frames per median


def longform_f0_stats(path: str) -> dict:
    """pyin F0 medians of one chunk (16 kHz, 60-450 Hz, 1024-sample frames, hop 320).

    ``f0_whole``: median F0 of the chunk's voiced frames. ``f0_start``/``f0_end``: median F0 in the first 2.5 s /
    last 3.5 s over the voiced frames within 0.67x-1.5x of ``f0_whole`` (drops creaky octave-down frames and pyin
    octave errors). Every median needs at least 10 frames, else it is None.
    """
    wav, _ = librosa.load(str(path), sr=_PYIN_SR, mono=True)
    f0, voiced, _ = librosa.pyin(wav, fmin=60, fmax=450, sr=_PYIN_SR, frame_length=1024, hop_length=_PYIN_HOP)
    t = np.arange(len(f0)) * _PYIN_HOP / _PYIN_SR
    dur = len(wav) / _PYIN_SR
    voiced = voiced & ~np.isnan(f0)

    def window_median(mask, lo, hi):
        m = mask & (t >= lo) & (t < hi)
        return float(np.median(f0[m])) if m.sum() >= _MIN_F0_FRAMES else None

    whole = float(np.median(f0[voiced])) if voiced.sum() >= _MIN_F0_FRAMES else None
    clean = voiced & (f0 >= 0.67 * whole) & (f0 <= 1.5 * whole) if whole else voiced & False
    return dict(
        dur=dur,
        f0_start=window_median(clean, 0, _JOIN_START_SEC),
        f0_end=window_median(clean, max(0, dur - _JOIN_END_SEC), dur),
        f0_whole=whole,
    )


def _join_jumps(stats: list) -> list:
    """|semitones| between the end of chunk i and the start of chunk i+1 (None when a median is missing)."""
    return [
        abs(float(12 * np.log2(b["f0_start"] / a["f0_end"]))) if b["f0_start"] and a["f0_end"] else None
        for a, b in zip(stats, stats[1:])
    ]


def compute_longform_chunk_metrics(
    generated_audio_dir: str,
    language: str = "en",
    asr_model_name: str = "nvidia/parakeet-tdt-1.1b",
    asr_model_type: str = "nemo",
    asr_batch_size: int = 32,
    strip_text_annotations_for_metrics: bool = False,
    device: str = "cuda",
    num_workers: int = 8,
    output_dir: Optional[str] = None,
) -> dict:
    """Per-chunk ASR and join pitch of a longform_chunked output directory.

    Reads ``<generated_audio_dir>/longform_chunks/`` (item_<i>_chunk_<k>.wav and item_<i>.json with each chunk's
    text), transcribes every chunk with the given ASR model and scores it with the dataset's text normalizer:
    per-chunk CER/WER, pooled CER/WER over all chunks (edits / reference length), and the number of chunks with
    CER above 3% and 10%. Join pitch jump: |semitones| between the median F0 of the last 3.5 s of a chunk and the
    first 2.5 s of the next chunk of the same item, over the voiced frames within 0.67x-1.5x of each chunk's median
    (longform_f0_stats); mean, median and % of joins where both medians exist.

    Writes ``longform_chunk_metrics.json`` (summary, chunks, joins) to ``output_dir`` (default:
    ``generated_audio_dir``) and returns it. Each chunk row keeps the raw ASR output as ``asr_text``.
    """
    chunk_dir = os.path.join(generated_audio_dir, LONGFORM_CHUNK_DIR)
    item_ids = sorted(int(f[5:-5]) for f in os.listdir(chunk_dir) if re.fullmatch(r"item_\d+\.json", f))
    chunks = []
    for item in item_ids:
        with open(os.path.join(chunk_dir, f"item_{item}.json"), encoding="utf-8") as f:
            info = json.load(f)
        for k, chunk in enumerate(info["chunks"]):
            path = os.path.join(chunk_dir, f"item_{item}_chunk_{k:03d}.wav")
            chunks.append(dict(item=item, chunk=k, file=os.path.basename(path), path=path, text=chunk["text"]))
    if not chunks:
        raise ValueError(f"No long-form chunks found in {chunk_dir}")
    paths = [c["path"] for c in chunks]

    # pitch on CPU workers (spawned when CUDA is already initialized in this process)
    start_time = time.time()
    context = "spawn" if torch.cuda.is_initialized() else "fork"
    with multiprocessing.get_context(context).Pool(max(1, num_workers)) as pool:
        f0 = pool.map(longform_f0_stats, paths, chunksize=1)
    logging.info(f"Long-form join pitch for {len(paths)} chunks in {time.time() - start_time:.0f}s")

    asr_model = load_asr_transcriber(asr_model_name, asr_model_type, device)
    asr_texts = asr_model.transcribe(audio_paths=paths, language=language, batch_size=asr_batch_size)
    del asr_model
    text_processor = get_text_processor(language)
    clean = strip_text_annotations_from_text if strip_text_annotations_for_metrics else (lambda x: x)
    refs = [text_processor.process_text_for_wer(clean(c["text"])) for c in chunks]
    hyps = [text_processor.process_text_for_wer(clean(h)) for h in asr_texts]

    for c, ref, hyp, asr_text, stats in zip(chunks, refs, hyps, asr_texts, f0):
        del c["path"]
        c.update(
            asr_text=asr_text,
            reference=ref,
            transcript=hyp,
            ref_chars=len(ref),
            cer=word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=True)[0],
            wer=word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=False)[0],
            **stats,
        )
    joins = []
    for item in item_ids:
        jumps = _join_jumps([c for c in chunks if c["item"] == item])
        joins += [dict(item=item, chunk=k, jump_semitones=j) for k, j in enumerate(jumps)]
    kept = [j["jump_semitones"] for j in joins if j["jump_semitones"] is not None]

    cers = [c["cer"] for c in chunks]
    summary = {
        "longform_chunks": len(chunks),
        "longform_chunk_cer_pooled": word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=True)[0],
        "longform_chunk_wer_pooled": word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)[0],
        "longform_chunk_cer_avg": float(np.mean(cers)),
        "longform_chunks_cer_gt_3pct": int(sum(c > 0.03 for c in cers)),
        "longform_chunks_cer_gt_10pct": int(sum(c > 0.10 for c in cers)),
    }
    summary.update(
        longform_joins=len(joins),
        longform_join_jump_mean=float(np.mean(kept)) if kept else float('nan'),
        longform_join_jump_median=float(np.median(kept)) if kept else float('nan'),
        longform_join_jump_kept_pct=100.0 * len(kept) / len(joins) if joins else float('nan'),
    )
    result = dict(summary=summary, chunks=chunks, joins=joins)
    output_dir = output_dir or generated_audio_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, LONGFORM_CHUNK_METRICS_FILE), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    logging.info(f"Long-form chunk metrics: {summary}")
    return result


def evaluate_longform(
    manifest_path,
    audio_dir,
    generated_audio_dir,
    language="en",
    sv_model_type="titanet",
    asr_model_name="nvidia/parakeet-tdt-1.1b",
    asr_model_type="nemo",
    with_utmosv2=True,
    strip_text_annotations_for_metrics=False,
    asr_batch_size=32,
    eou_batch_size=32,
    device="cuda",
    eou_model_name=None,
    output_dir=None,
):
    """evaluate() for longform_chunked outputs, without loading any whole long file into an ASR, SV or EoU model.

    Files can be an hour long, which whole-file ASR (and the SV / EoU models) cannot hold in GPU memory. The chunk
    files under longform_chunks/ are scored instead (compute_longform_chunk_metrics, the only ASR pass):

    - cer / wer (filewise, cumulative): hypothesis = the chunk transcripts of a file joined in order, scored against
      the file's full reference text with the usual normalizer; same keys and meaning as evaluate().
    - ground-truth-audio ASR (cer_pred_gt_audio*, cer_gt_audio_cumulative, ...): would need whole-file ASR of the
      target audio, so not computed (NaN).
    - speaker similarity: whole-file keys (ssim_pred_context_avg, ssim_pred_gt_avg, ...) are NaN. Instead
      ``ssim_pred_context_chunk_avg`` (+ ``_alternate``): per file, the chunk-duration-weighted mean of the cosine
      similarity between each chunk and the context audio, averaged over files.
    - utmosv2_avg: whole file as before (UTMOSv2 scores a fixed-length crop, so long files are memory-safe).
    - EoU rates: classification of the file's last chunk against that chunk's text (the end of the file).
    - FCD and prosody metrics: not computed.
    The longform_* chunk summary (per-chunk ASR, join pitch) is added as well.
    The chunk metrics JSON goes to ``output_dir`` (default: ``generated_audio_dir``).

    Returns:
        Tuple of (avg_metrics dict, filewise_metrics list), as evaluate().
    """
    start_time = time.time()
    chunk_result = compute_longform_chunk_metrics(
        generated_audio_dir,
        language=language,
        asr_model_name=asr_model_name,
        asr_model_type=asr_model_type,
        asr_batch_size=asr_batch_size,
        strip_text_annotations_for_metrics=strip_text_annotations_for_metrics,
        device=device,
        output_dir=output_dir,
    )
    audio_file_lists = find_generated_audio_files(generated_audio_dir)
    records = read_manifest(manifest_path)
    assert len(audio_file_lists) == len(records)
    chunk_dir = os.path.join(generated_audio_dir, LONGFORM_CHUNK_DIR)
    chunks_of = {}
    for c in chunk_result["chunks"]:
        chunks_of.setdefault(c["item"], []).append(c)

    text_processor = get_text_processor(language)
    clean = strip_text_annotations_from_text if strip_text_annotations_for_metrics else (lambda x: x)
    models = load_speaker_verification_models(sv_model_type, device)
    utmosv2_scores = compute_utmosv2_scores(generated_audio_dir, device) if with_utmosv2 else None
    if with_utmosv2 and not UTMOSV2_AVAILABLE:
        logging.warning("UTMOSv2 was requested but the UTMOSv2 library is not available; scores are NaN.")

    rows, eou_items = [], []
    for ridx, record in enumerate(records):
        file_chunks = chunks_of[ridx]
        chunk_paths = [os.path.join(chunk_dir, c["file"]) for c in file_chunks]
        text_field = next((k for k in ("original_text", "normalized_text") if k in record), "text")
        gt_text = text_processor.process_text_for_wer(clean(record[text_field]))
        pred_text = text_processor.process_text_for_wer(clean(" ".join(c["asr_text"] for c in file_chunks)))
        detailed_cer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=True)
        detailed_wer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=False)
        eou_items.append((chunk_paths[-1], text_processor.process_text_for_wer(clean(file_chunks[-1]["text"]))))

        context_audio_filepath = _resolve_path(audio_dir, record.get('context_audio_filepath'))
        durations = np.array([get_wav_file_duration(p) for p in chunk_paths])
        ssim = {}
        for key, model in (("pred_context_ssim_chunks", "sv_model"), ("pred_context_ssim_chunks_alternate", None)):
            embed = partial(
                extract_embedding,
                model=models[model] if model else models["sv_model_alternate"],
                extractor=models["feature_extractor"],
                device=device,
                sv_model_type=sv_model_type if model else "titanet",
            )
            with torch.inference_mode():
                context_embedding = embed(audio_path=context_audio_filepath)
                sims = [
                    torch.nn.functional.cosine_similarity(embed(audio_path=p), context_embedding, dim=0).item()
                    for p in chunk_paths
                ]
            ssim[key] = float(np.sum(durations * np.array(sims)) / durations.sum())

        pred_audio_filepath = audio_file_lists[ridx]
        nan = float('nan')
        rows.append(
            {
                'gt_text': gt_text,
                'pred_text': pred_text,
                'gt_audio_text': None,
                'detailed_cer': detailed_cer,
                'detailed_wer': detailed_wer,
                'cer': detailed_cer[0],
                'cer_pred_gt_audio': nan,
                'wer': detailed_wer[0],
                'wer_pred_gt_audio': nan,
                'pred_gt_ssim': nan,
                'pred_context_ssim': nan,
                'gt_context_ssim': nan,
                'pred_gt_ssim_alternate': nan,
                'pred_context_ssim_alternate': nan,
                'gt_context_ssim_alternate': nan,
                **ssim,
                'gt_audio_filepath': _resolve_path(audio_dir, record.get('audio_filepath')),
                'pred_audio_filepath': pred_audio_filepath,
                'context_audio_filepath': context_audio_filepath,
                'utmosv2': (utmosv2_scores.get(os.path.normpath(pred_audio_filepath), nan) if utmosv2_scores else nan),
                'total_gen_audio_seconds': get_wav_file_duration(pred_audio_filepath),
                'num_chunks': len(file_chunks),
            }
        )

    eou_results = None
    if language == "en":  # EoU supports English only, as in evaluate_dir
        eou_kwargs = {"device": device}
        if eou_model_name is not None:
            eou_kwargs["model_name"] = eou_model_name
        eou_results = classify_eou_batched(EoUClassifier(**eou_kwargs), eou_items, batch_size=eou_batch_size)
    for ridx, row in enumerate(rows):
        result = eou_results[ridx] if eou_results is not None else None
        row['eou_type'] = result.eou_type.value if result else None
        row['eou_trailing_duration'] = result.trailing_duration if result else float('nan')
        row['eou_trail_rms_ratio'] = result.trail_rms_ratio if result else float('nan')

    avg_metrics = compute_global_metrics(filewise_metrics=rows, device=device)
    n = len(rows)
    avg_metrics['ssim_pred_context_chunk_avg'] = sum(r['pred_context_ssim_chunks'] for r in rows) / n
    avg_metrics['ssim_pred_context_chunk_avg_alternate'] = (
        sum(r['pred_context_ssim_chunks_alternate'] for r in rows) / n
    )
    avg_metrics.update(chunk_result["summary"])
    logging.info(f"evaluate_longform() completed in {time.time() - start_time:.1f}s")
    filewise_metrics_to_save = list(FILEWISE_METRICS_TO_SAVE) + LONGFORM_FILEWISE_METRICS_TO_SAVE
    return avg_metrics, [{k: m[k] for k in filewise_metrics_to_save if k in m} for m in rows]


def compute_fcd(gt_audio_paths, predicted_codes_paths, codec_model_path, device="cuda"):
    """Compute Frechet Codec Distance from ground-truth audio paths and predicted codec codes paths.

    Args:
        gt_audio_paths: List of paths to ground-truth audio files.
        predicted_codes_paths: List of paths to predicted codec codes (.pt) files.
        codec_model_path: Path or name of the codec model for FCD computation.
        device: Device to use for computation.

    Returns:
        FCD score (float).
    """
    fcd_metric = FrechetCodecDistance(codec_name=codec_model_path).to(device)
    for gt_path, codes_path in zip(gt_audio_paths, predicted_codes_paths):
        fcd_metric.update_from_audio_file(gt_path, True)
        predicted_codes = torch.load(codes_path).unsqueeze(0).to(device)  # B, C, T
        predicted_codes_lens = torch.tensor([predicted_codes.size(-1)], dtype=torch.int, device=device)
        fcd_metric.update(predicted_codes, predicted_codes_lens, False)
    fcd = fcd_metric.compute().cpu().item()
    fcd_metric.reset()
    return fcd


def _mean_finite_metric(filewise_metrics, key: str) -> float:
    values = []
    for metrics in filewise_metrics:
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return float('nan') if not values else float(np.mean(values))


def compute_global_metrics(
    filewise_metrics,
    gt_audio_paths=None,
    predicted_codes_paths=None,
    codec_model_path=None,
    device="cuda",
):
    """Aggregate metrics from per-file results.

    Args:
        filewise_metrics: List of per-file metric dicts. Each must contain at least
            'pred_text', 'gt_text', 'cer', 'wer', 'pred_gt_ssim', 'pred_context_ssim',
            'gt_context_ssim', 'utmosv2'.
        gt_audio_paths: Optional list of ground-truth audio paths for FCD computation.
        predicted_codes_paths: Optional list of predicted codec codes paths for FCD computation.
        codec_model_path: Optional codec model path/name for FCD computation.

    Returns:
        dict of global metrics (same keys as evaluate() avg_metrics).
    """
    n = len(filewise_metrics)
    pred_texts = [m['pred_text'] for m in filewise_metrics]
    gt_texts = [m['gt_text'] for m in filewise_metrics]

    avg_metrics = {}
    avg_metrics['cer_filewise_avg'] = sum(m['cer'] for m in filewise_metrics) / n
    avg_metrics['wer_filewise_avg'] = sum(m['wer'] for m in filewise_metrics) / n
    avg_metrics['cer_cumulative'] = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=True)[0]
    avg_metrics['wer_cumulative'] = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=False)[
        0
    ]
    # Japanese reading-based CER (Katakana via pyopenjtalk); only present for ja datasets.
    kata = [m for m in filewise_metrics if m.get('katakana_cer') is not None]
    if kata:
        avg_metrics['katakana_cer_filewise_avg'] = sum(m['katakana_cer'] for m in kata) / len(kata)
        avg_metrics['katakana_cer_cumulative'] = word_error_rate_detail(
            hypotheses=[m['pred_katakana'] for m in kata],
            references=[m['gt_katakana'] for m in kata],
            use_cer=True,
        )[0]
    avg_metrics['ssim_pred_gt_avg'] = sum(m['pred_gt_ssim'] for m in filewise_metrics) / n
    avg_metrics['ssim_pred_context_avg'] = sum(m['pred_context_ssim'] for m in filewise_metrics) / n
    avg_metrics['ssim_gt_context_avg'] = sum(m['gt_context_ssim'] for m in filewise_metrics) / n
    avg_metrics['ssim_pred_gt_avg_alternate'] = sum(m['pred_gt_ssim_alternate'] for m in filewise_metrics) / n
    avg_metrics['ssim_pred_context_avg_alternate'] = (
        sum(m['pred_context_ssim_alternate'] for m in filewise_metrics) / n
    )
    avg_metrics['ssim_gt_context_avg_alternate'] = sum(m['gt_context_ssim_alternate'] for m in filewise_metrics) / n
    if 'pred_gt_esim' in filewise_metrics[0]:
        avg_metrics['esim_pred_gt_avg'] = sum(m['pred_gt_esim'] for m in filewise_metrics) / n
        avg_metrics['ems_pred_gt_avg'] = sum(m['pred_gt_ems'] for m in filewise_metrics) / n
    if 'pitch_distance' in filewise_metrics[0]:
        avg_metrics['pitch_distance_avg'] = _mean_finite_metric(filewise_metrics, 'pitch_distance')
        avg_metrics['intensity_distance_avg'] = _mean_finite_metric(filewise_metrics, 'intensity_distance')
        avg_metrics['speech_rate_distance_avg'] = _mean_finite_metric(filewise_metrics, 'speech_rate_distance')

    # Cumulative WER/CER on ground-truth audio transcriptions (if available)
    gt_audio_texts = [m['gt_audio_text'] for m in filewise_metrics]
    avg_metrics['cer_pred_gt_audio_filewise_avg'] = _mean_finite_metric(filewise_metrics, 'cer_pred_gt_audio')
    avg_metrics['wer_pred_gt_audio_filewise_avg'] = _mean_finite_metric(filewise_metrics, 'wer_pred_gt_audio')
    if None not in gt_audio_texts:
        avg_metrics['cer_pred_gt_audio_cumulative'] = word_error_rate_detail(
            hypotheses=pred_texts, references=gt_audio_texts, use_cer=True
        )[0]
        avg_metrics['wer_pred_gt_audio_cumulative'] = word_error_rate_detail(
            hypotheses=pred_texts, references=gt_audio_texts, use_cer=False
        )[0]
        avg_metrics['cer_gt_audio_cumulative'] = word_error_rate_detail(
            hypotheses=gt_audio_texts, references=gt_texts, use_cer=True
        )[0]
        avg_metrics['wer_gt_audio_cumulative'] = word_error_rate_detail(
            hypotheses=gt_audio_texts, references=gt_texts, use_cer=False
        )[0]
    else:
        avg_metrics['cer_pred_gt_audio_cumulative'] = float('NaN')
        avg_metrics['wer_pred_gt_audio_cumulative'] = float('NaN')
        avg_metrics['cer_gt_audio_cumulative'] = float('NaN')
        avg_metrics['wer_gt_audio_cumulative'] = float('NaN')
        logging.warning(
            "Ground truth audio files are missing. Setting cumulative CER and WER for ground truth audio to NaN."
        )

    avg_metrics['utmosv2_avg'] = sum(m['utmosv2'] for m in filewise_metrics) / n
    avg_metrics['total_gen_audio_seconds'] = sum(m['total_gen_audio_seconds'] for m in filewise_metrics)

    # EoU classification rates
    eou_types = [m.get('eou_type') for m in filewise_metrics]
    if eou_types[0] is not None:
        eou_counts = Counter(eou_types)
        for label in EoUType.error_types():
            avg_metrics[f'eou_{label}_rate'] = eou_counts.get(label, 0) / n
        # Aggregate error rate: fraction of all non-GOOD cases
        avg_metrics['eou_error_rate'] = 1.0 - eou_counts.get(EoUType.GOOD, 0) / n
    else:
        for label in EoUType.error_types():
            avg_metrics[f'eou_{label}_rate'] = float('nan')
        avg_metrics['eou_error_rate'] = float('nan')

    # FCD: compute only if all required paths are provided
    if gt_audio_paths and predicted_codes_paths and codec_model_path:
        avg_metrics['frechet_codec_distance'] = compute_fcd(
            gt_audio_paths, predicted_codes_paths, codec_model_path, device=device
        )
    else:
        avg_metrics['frechet_codec_distance'] = float('nan')

    pprint.pprint(avg_metrics)
    return avg_metrics


def main():
    # audio_dir="/datap/misc/Datasets/riva" \
    parser = argparse.ArgumentParser(description='Evaluate Generated Audio')
    parser.add_argument('--manifest_path', type=str, default=None)
    parser.add_argument('--audio_dir', type=str, default=None)
    parser.add_argument('--generated_audio_dir', type=str, default=None)
    parser.add_argument('--language', type=str, default="en")
    parser.add_argument('--evalset', type=str, default=None)
    parser.add_argument(
        '--with_prosody_metrics',
        action='store_true',
        help='Compute ESIM/EMS and pitch, intensity, and speech-rate distance metrics.',
    )
    parser.add_argument('--prosody_model_size', type=str, default="small", choices=["small", "large"])
    parser.add_argument(
        '--strip_text_annotations_for_metrics',
        action='store_true',
        help='Strip bracket/tag/control annotations from reference and ASR hypothesis text while computing text metrics.',
    )
    parser.add_argument('--asr_model_name', type=str, default="nvidia/parakeet-tdt-1.1b")
    parser.add_argument('--asr_model_type', type=str, default="nemo", choices=["nemo", "nemo_with_prompt", "whisper"])
    parser.add_argument('--asr_batch_size', type=int, default=32)
    parser.add_argument(
        '--longform_rescore',
        action='store_true',
        help='Re-score a longform_chunked output end to end with evaluate_longform (file-level metrics from the chunk '
        'files + chunk metrics): --generated_audio_dir is audio/repeat_<r>; --manifest_path and --audio_dir default '
        'to its longform_manifest.jsonl and itself. Writes metrics.json, filewise_metrics.json and '
        'longform_chunk_metrics.json to --output_dir.',
    )
    parser.add_argument('--output_dir', type=str, default=None, help='Where --longform_rescore writes its JSON.')
    parser.add_argument('--eou_batch_size', type=int, default=32, help='EoU batch size for --longform_rescore.')
    args = parser.parse_args()

    if args.longform_rescore:
        output_dir = args.output_dir or args.generated_audio_dir
        metrics, filewise = evaluate_longform(
            manifest_path=args.manifest_path or os.path.join(args.generated_audio_dir, "longform_manifest.jsonl"),
            audio_dir=args.audio_dir or args.generated_audio_dir,
            generated_audio_dir=args.generated_audio_dir,
            language=args.language,
            asr_model_name=args.asr_model_name,
            asr_model_type=args.asr_model_type,
            asr_batch_size=args.asr_batch_size,
            eou_batch_size=args.eou_batch_size,
            strip_text_annotations_for_metrics=args.strip_text_annotations_for_metrics,
            output_dir=output_dir,
        )
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(output_dir, "filewise_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(filewise, f, indent=4, ensure_ascii=False)
        return

    if args.evalset is not None:
        dataset_meta_info = load_evalset_config()
        assert args.evalset in dataset_meta_info, f"Dataset '{args.evalset}' not found in evalset_config.json"
        args.manifest_path = dataset_meta_info[args.evalset]['manifest_path']
        args.audio_dir = dataset_meta_info[args.evalset]['audio_dir']

    evaluate(
        args.manifest_path,
        args.audio_dir,
        args.generated_audio_dir,
        args.language,
        sv_model_type="wavlm",
        asr_model_name="nvidia/parakeet-ctc-0.6b",
        with_prosody_metrics=args.with_prosody_metrics,
        strip_text_annotations_for_metrics=args.strip_text_annotations_for_metrics,
        prosody_model_size=args.prosody_model_size,
    )


if __name__ == "__main__":
    main()
