# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Tests for the text chunking and configuration of --easy_magpie_inference_mode longform_chunked."""

import pytest
import torch

from nemo.collections.tts.modules.magpietts_inference.inference import (
    EasyMagpieLongformInferenceConfig,
    EasyMagpieLongformInferenceRunner,
    chunk_text_by_words,
    decode_with_codec_history,
    split_sentences,
)


def _sentence(n_words: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n_words - 1)) + f" {tag}end."


class TestLongformChunking:
    @pytest.mark.unit
    def test_split_sentences_keeps_abbreviations_and_initials(self):
        text = 'Dr. Smith met J. Doe at noon.  "Really?" she asked! Then they left'
        assert split_sentences(text) == ['Dr. Smith met J. Doe at noon.', '"Really?"', 'she asked!', 'Then they left']

    @pytest.mark.unit
    def test_chunks_are_whole_sentences_within_range(self):
        sentences = [_sentence(n, f"s{k}w") for k, n in enumerate([12, 25, 18, 9, 30, 15, 22, 11, 27])]
        text = " ".join(sentences)
        chunks = chunk_text_by_words(text, 40, 60)
        assert " ".join(chunks) == text
        assert all(40 <= len(c.split()) <= 60 for c in chunks)
        assert all(c.endswith("end.") for c in chunks)  # no sentence is split

    @pytest.mark.unit
    def test_long_sentence_is_cut_at_last_comma_within_limit(self):
        words = [f"w{i}," if i == 49 else f"w{i}" for i in range(70)]
        chunks = chunk_text_by_words(" ".join(words) + ".", 10, 60)
        assert chunks[0].split()[-1] == "w49,"
        assert sum(len(c.split()) for c in chunks) == 70


class TestLongformConfig:
    @pytest.mark.unit
    def test_defaults_and_silence_steps(self):
        config = EasyMagpieLongformInferenceConfig(batch_size=1)
        assert (config.chunk_min_words, config.chunk_max_words, config.carry) == (40, 60, "full_audio_only")
        assert config.reference_bias == 0.0 and not config.force_espeak_ipa and config.eos_detection_method is None
        assert config.user_silence_steps == 5 and config.codec_warm
        assert EasyMagpieLongformInferenceConfig(batch_size=1, carry="tail").user_silence_steps == 4

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(carry="previous"),
            dict(eos_detection_method="argmax_any"),
            dict(chunk_min_words=0),
        ],
    )
    def test_invalid_settings_raise(self, kwargs):
        with pytest.raises(ValueError):
            EasyMagpieLongformInferenceConfig(batch_size=1, **kwargs)


class _CausalCodec:
    """Two samples per frame: the frame's code, then the code of the frame before it (-1 for the first frame)."""

    def codes_to_audio(self, codes, codes_lens):
        c = codes[0, 0].float()
        previous = torch.cat([torch.tensor([-1.0]), c[:-1]])
        return torch.stack([c, previous], dim=-1).reshape(1, -1), codes_lens * 2, codes


class TestCodecHistoryDecode:
    @pytest.mark.unit
    def test_chunk_is_decoded_after_history_and_trimmed(self):
        wave = decode_with_codec_history(_CausalCodec(), torch.tensor([[0, 0]]), torch.tensor([[7, 8]]))
        # the first frame sees a silence frame instead of a cold start
        assert wave.tolist() == [7.0, 0.0, 8.0, 7.0]


class _WarmModel:
    """Model stub: speech delay 2 steps x frame stacking 2 = 4 silence frames; silence code 0 in both codebooks."""

    frame_stacking_factor = 2
    training_modes = [type("Mode", (), {"streaming_speech_delay": 2})()]
    _codec_sil_codes_buffer = torch.tensor([0, 0])
    codec_sil_codes = _codec_sil_codes_buffer
    device = "cpu"


def _warm_runner(**kwargs):
    runner = object.__new__(EasyMagpieLongformInferenceRunner)
    runner.model, runner.config = _WarmModel(), EasyMagpieLongformInferenceConfig(batch_size=1, **kwargs)
    return runner


class TestCodecWarmHistory:
    @pytest.mark.unit
    def test_history_is_speech_delay_silence_frames(self):
        history = _warm_runner()._codec_warm_history(torch.tensor([[7, 8], [9, 9]]))
        assert history.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]

    @pytest.mark.unit
    def test_off_and_empty_chunk_decode_alone(self):
        assert _warm_runner(codec_warm=False)._codec_warm_history(torch.tensor([[7], [9]])) is None
        assert _warm_runner()._codec_warm_history(torch.zeros(2, 0, dtype=torch.long)) is None
