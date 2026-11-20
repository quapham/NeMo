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

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.magpietts_modules import LocalTransformerHelper, SpecialAudioToken

CODEBOOK_SIZE, C, S, E = 8, 2, 2, 4  # 4 local-transformer indices per frame
VOCAB = CODEBOOK_SIZE + len(SpecialAudioToken)
EOS = SpecialAudioToken.get_index(SpecialAudioToken.AUDIO_EOS, CODEBOOK_SIZE)


class _Identity(nn.Module):
    def reset_cache(self, use_cache=False):
        pass

    def forward(self, x, mask):
        return {'output': x}


class _EosHead(nn.Module):
    """Output head whose EOS logit is +/-100 (EOS certain or impossible) and whose audio logits are flat."""

    def __init__(self, eos: bool):
        super().__init__()
        self.eos = eos

    def forward(self, x):
        logits = torch.zeros(x.size(0), VOCAB)
        logits[:, EOS] = 100.0 if self.eos else -100.0
        return logits


def _helper(eos_indices):
    return LocalTransformerHelper(
        local_transformer=_Identity(),
        audio_embeddings=nn.ModuleList([nn.Embedding(VOCAB, E) for _ in range(C * S)]),
        audio_in_projection=nn.Identity(),
        local_transformer_in_projection=nn.Identity(),
        local_transformer_audio_out_projection=nn.Identity(),
        local_transformer_out_projections=nn.ModuleList([_EosHead(i in eos_indices) for i in range(C * S)]),
        num_audio_codebooks=C,
        frame_stacking_factor=S,
        audio_eos_id=EOS,
        mask_token_id=SpecialAudioToken.get_index(SpecialAudioToken.MASK_TOKEN, CODEBOOK_SIZE),
        codebook_size=CODEBOOK_SIZE,
    )


@pytest.mark.unit
@pytest.mark.parametrize("use_cfg", [False, True])
@pytest.mark.parametrize(
    "eos_indices, frame_is_eos",
    [({0, 1, 2}, True), ({0, 1, 2, 3}, True), ({0, 1}, False), ({3}, False), (set(), False)],
)
def test_majority_eos_frame_is_all_eos_or_eos_free(eos_indices, frame_is_eos, use_cfg):
    torch.manual_seed(0)
    batch = 2 if use_cfg else 1  # with CFG the batch holds conditional + unconditional rows
    codes = _helper(eos_indices).sample_autoregressive(
        torch.zeros(batch, E), use_cfg=use_cfg, cfg_scale=1.0, use_kv_cache=False, topk=4, majority_eos=True
    )
    assert codes.shape == (1, C, S)
    assert bool((codes == EOS).all()) if frame_is_eos else not bool((codes == EOS).any())


@pytest.mark.unit
def test_default_rule_keeps_minority_eos():
    codes = _helper({3}).sample_autoregressive(torch.zeros(1, E), use_kv_cache=False, topk=4)
    assert int((codes == EOS).sum()) == 1


@pytest.mark.unit
def test_majority_and_zero_cb_are_exclusive():
    with pytest.raises(ValueError):
        _helper(set()).sample_autoregressive(torch.zeros(1, E), zero_cb_eos=True, majority_eos=True)
