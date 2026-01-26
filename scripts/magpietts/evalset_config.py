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
dataset_meta_info = {
    'riva_hard_digits': {
        'manifest_path': '/Data/evaluation_manifests/hard-digits-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_letters': {
        'manifest_path': '/Data/evaluation_manifests/hard-letters-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_money': {
        'manifest_path': '/Data/evaluation_manifests/hard-money-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_short': {
        'manifest_path': '/Data/evaluation_manifests/hard-short-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'vctk': {
        'manifest_path': '/Data/evaluation_manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withcontextaudiopaths_silence_trimmed.json',
        'audio_dir': '/Data/VCTK-Corpus-0.92',
        'feature_dir': '/Data/VCTK-Corpus-0.92',
    },
    'libritts_seen': {
        'manifest_path': '/Data/evaluation_manifests/LibriTTS_seen_evalset_from_testclean_v2.json',
        'audio_dir': '/Data/LibriTTS',
        'feature_dir': '/Data/LibriTTS',
    },
    'libritts_test_clean': {
        'manifest_path': '/Data/evaluation_manifests/LibriTTS_test_clean_withContextAudioPaths.jsonl',
        'audio_dir': '/Data/LibriTTS',
        'feature_dir': '/Data/LibriTTS',
    },
    # We need an4_val_ci just for CI tests
    'an4_val_ci': {
        'manifest_path': '/home/TestData/an4_dataset/an4_val_context_v1.json',
        'audio_dir': '/',
        'feature_dir': None,
    },'vietnamese_north_female': {
        'manifest_path' : '/output/val_north_female_v1_original.json',
        'audio_dir': '/',
        'feature_dir': '/',
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },'vietnamese_north_female_long_context': {
        'manifest_path' : '/output/north_female_with_long_context.json',
        'audio_dir': '/',
        'feature_dir': '/',
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },'vietnamese_long_nf_context': {
        'manifest_path' : '/output/long_speaker_with_nf_context.json',
        'audio_dir': '/',
        'feature_dir': '/',
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },'vietnamese_unseen_multispeaker': {
        'manifest_path' : '/output/unseen_multispeaker_vi.json',
        'audio_dir': '/',
        'feature_dir': '/',
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },'vietnamese_vivos_dur3to8s': {
        'manifest_path' : '/data/vivos/vivos_dur3to8s.jsonl',
        'audio_dir': '/',
        'feature_dir': '/',
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },'vietnamese_long_vivos': {
        'manifest_path' : '/output/vivos_with_long_context.jsonl',
        'audio_dir': '/',
        'feature_dir': '/',
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },'vietnamese_long': {
        'manifest_path' : '/output/manifest_long.json',
        'audio_dir': '/data/Long',
        'feature_dir': '/',
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },'vietnamese_long_vivos_text_context': {
        'manifest_path' : '/output/vivos_with_long_text_context.jsonl',
        'audio_dir': '/',
        'feature_dir': '/',
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },
    # New manifest files from 11_3_magpietts_2508 with updated context texts
    'en_emma_additional': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_emma_additional.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_lindy_other': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_lindy_other.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_megan_additional': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_megan_additional.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_rodney_other': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_rodney_other.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_sean_additional': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_sean_additional.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_tom_additional': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_tom_additional.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_carlos_regular': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_carlos_regular.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_rubby_regular': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_rubby_regular.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_samy_neutral': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_samy_neutral.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_virginie_regular': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_virginie_regular.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_houzhen_regular': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_houzhen_regular.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },
    'en_siwei_regular': {
        'manifest_path': '/output/manifest_11_3_rel/manifest_11_3_rel_dummy/en_siwei_regular.json',
        'audio_dir': "/",
        'feature_dir': '',
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
    },'hi_ai4bharat_espeak':{
        'manifest_path': '/data/hindi_data/datasets/hi_val_manifest_17z_phonemizer.json',
        'audio_dir': '/data/hindi_data',
        'feature_dir': None,
        'tokenizer_names': ['hindi_phoneme'],
        'whisper_language': 'hi',
        'load_cached_codes_if_available': False
    },
    'hi_ai4bharat':{
        'manifest_path': '/output/hi_val_manifest.jsonl',
        'audio_dir': '/data/hindi_data',
        'feature_dir': None,
        'tokenizer_names': ['hindi_phoneme'],
        'whisper_language': 'hi',
        'load_cached_codes_if_available': False
    },
    'hi_ai4bharat_char':{
        'manifest_path': '/output/hi_val_manifest.jsonl',
        'audio_dir': '/data/hindi_data',
        'feature_dir': None,
        'tokenizer_names': ['hindi_chartokenizer'],
        'whisper_language': 'hi',
        'load_cached_codes_if_available': False
    },
    'vi_multi_unseen':{
        'manifest_path': '/output/vivos_dur5s_manifest_norm.jsonl',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },
    'ja_multi_unseen':{
        'manifest_path': '/output/japanese_eval_manifest/v3_jvs_jsut.json',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['japanese_phoneme'],
        'whisper_language': 'ja',
        'load_cached_codes_if_available': False
    },'vi_long_context_audio':{
        'manifest_path': '/output/manifest_long.json',
        'audio_dir': '/data/Long',
        'feature_dir': None,
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },
    'vi_unseen_speaker_long_context_text':{
        'manifest_path': '/output/vivos_with_long_context_text.jsonl',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },'vi_unseen_speaker_long_context_audio':{
        'manifest_path': '/output/vivos_with_long_context_audio.jsonl',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },
    'vi_unseen_speaker_phung_context_audio':{
        'manifest_path': '/output/vivos_with_phung_context_audio.jsonl',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },'vi_unseen_speaker_phung_context_text':{
        'manifest_path': '/output/vivos_with_phung_context_text.jsonl',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },
    'grpo_en':{
        'manifest_path': '/output/grpo_val/en_val_text_context_pairs_aligned.json',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['english_phoneme'],
        'whisper_language': 'en',
        'load_cached_codes_if_available': False
    },'grpo_es':{
        'manifest_path': '/output/grpo_val/es_val_text_context_pairs_aligned.json',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es',
        'load_cached_codes_if_available': False
    },'grpo_de':{
        'manifest_path': '/output/grpo_val/de_val_text_context_pairs_aligned.json',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },'grpo_fr':{
        'manifest_path': '/output/grpo_val/fr_val_text_context_pairs_aligned.json',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
        'load_cached_codes_if_available': False
    },'grpo_vi':{
        'manifest_path': '/output/grpo_val/vi_val_grpo_manifest.jsonl',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['vietnamese_phoneme'],
        'whisper_language': 'vi',
        'load_cached_codes_if_available': False
    },'grpo_hi':{
        'manifest_path': '/output/grpo_val/hi_val_grpo_manifest.jsonl',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['hindi_chartokenizer'],
        'whisper_language': 'hi',
        'load_cached_codes_if_available': False
    },'grpo_ja':{
        'manifest_path': '/output/grpo_val/ja_val_grpo_manifest.jsonl',
        'audio_dir': '/',
        'feature_dir': None,
        'tokenizer_names': ['japanese_phoneme'],
        'whisper_language': 'ja',
        'load_cached_codes_if_available': False
    },
    # CML test
    'spanish_cml_phoneme': {
        'manifest_path' : '/output/multi_eval/spanish/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es'
    },
    'german_cml_phoneme': {
        'manifest_path' : '/output/multi_eval/german/test_withAudioCodes_codec21Khz_no_eliz_filtered_100_samples.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },
    'french_cml_sep_char': {
        'manifest_path' : '/output/multi_eval/french/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
        'load_cached_codes_if_available': False
    },
    'italian_cml_sep_char': {
        'manifest_path' : '/output/multi_eval/italian/test_withAudioCodes_codec21Khz_no_eliz_filtered_100_samples.json',
        'feature_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'tokenizer_names': ['italian_phoneme'],
        'whisper_language': 'it',
        'load_cached_codes_if_available': False
    },
    'mandarin_cml_sep_char': {
        'manifest_path' : '/output/multi_eval/mandarin/test_withAudioCodes_codec21Khz_mandarin_chinese_text_100_samples.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'tokenizer_names': ['mandarin_phoneme'],
        'whisper_language': 'zh',
        'load_cached_codes_if_available': False
    },

    # punct
    'spanish_cml_phoneme_punct': {
        'manifest_path' : '/output/multi_eval/spanish/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset_punc.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es'
    },
    'german_cml_phoneme_punct': {
        'manifest_path' : '/output/multi_eval/german/test_withAudioCodes_codec21Khz_no_eliz_filtered_100_samples_punc.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },
    'french_cml_sep_char_punct': {
        'manifest_path' : '/output/multi_eval/french/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset_punc.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
        'load_cached_codes_if_available': False
    },
    'italian_cml_sep_char_punct': {
        'manifest_path' : '/output/multi_eval/italian/test_withAudioCodes_codec21Khz_no_eliz_filtered_100_samples_punc.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'tokenizer_names': ['italian_phoneme'],
        'whisper_language': 'it',
        'load_cached_codes_if_available': False
    }, 


    'libri_dev_clean_eval_large': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets/manifests/t5_exp/dev_clean_withContextAudioPaths_withTargetCodes_evalset_large.json',
        'audio_dir' : '/datap/misc/Data/LibriTTS',
        'feature_dir' : '/datap/misc/Data/LibriTTS',
    }, 
    # Lindy and Rodney, text context, validation set
    'riva_val_text_context': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets/manifests/t5_exp/Mikyas_modified_RivattsEnglishLindyRodney21fps_val_nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_phoneme_tts_TextContext.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },
    # same as what was known as "riva_challenging" but Paarth removed some problematic items
    'riva_challenging_filtered': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets/manifests/t5_exp/from_paarth/riva_challenging_filtered.json',
        'audio_dir' : '/datap/misc/Data/riva',
        'feature_dir' : '',
    },
    'spanish_cml_phoneme_md': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/manifests_with_codecs/cml_tts_dataset_spanish_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/datap/misc/Data/CML/cml_tts_dataset_spanish_v0.1/',
        'feature_dir': '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
        'load_cached_codes_if_available': False
    },
    'french_cml_sep_char_md': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/manifests_with_codecs/cml_tts_dataset_french_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/datap/misc/Data/CML/cml_tts_dataset_french_v0.1/',
        'feature_dir': '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
        'load_cached_codes_if_available': False
    },
    'german_cml_sep_phoneme_md': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/manifests_with_codecs/cml_tts_dataset_german_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/datap/misc/Data/CML/cml_tts_dataset_german_v0.1/',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },
    'mandarin_sep_phoneme_md': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/manifests_with_codecs/mandarin/test_withAudioCodes_codec21Khz_mandarin_chinese_text.json',
        'audio_dir': '/datap/misc/Data/CML/cml_tts_dataset_german_v0.1/',
        'feature_dir': '',
        'tokenizer_names': ['mandarin_phoneme'],
        'whisper_language': 'zh',
        'load_cached_codes_if_available': False
    },
    'italian_cml_sep_char_md': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/manifests_with_codecs/cml_tts_dataset_italian_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/datap/misc/Data/CML/cml_tts_dataset_italian_v0.1/',
        'feature_dir': '',
        'tokenizer_names': ['italian_phoneme'],
        'whisper_language': 'it',
        'load_cached_codes_if_available': False
    },
    'door_dash_all': {
        'manifest_path' : '/home/lustre/fs12/portfolios/edgeai/users/mdesta/utils/doordash_space_only_inference_eval_set_400mins_manifest_fully_normalized.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'door_dash': {
        'manifest_path' : '/home/lustre/fs12/portfolios/edgeai/users/mdesta/utils/doordash_no_empty_inference_eval_set_400mins_manifest_fully_normalized.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'lindy_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/lindy_riva_subjective__wizwiki_lindy_rodney_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'rodney_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/rodney_riva_subjective__wizwiki_lindy_rodney_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'emma_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/generated_json_files/emma_riva_subjective.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'sean_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/generated_json_files/sean_riva_subjective.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'tom_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/generated_json_files/tom_riva_subjective.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'megan_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/generated_json_files/megan_riva_subjective.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'rubby_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/en/rubby_riva_subjective_en_text_es_female_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'carlos_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/en/carlos_riva_subjective_en_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'virginie_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/en/virginie_riva_subjective_en_text_fr_female_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'sammy_en_iva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/en/samy_riva_subjective_en_text_fr_male_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'rubby_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/es/rubby_riva_subjective_es_female_es_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    }, 
    'carlos_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/es/carlos_riva_subjective_es_male_es_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    },  
    'lindy_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/es/lindy_riva_subjective_es_female_en_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    }, 
    'rodney_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/es/rodney_riva_subjective_es_male_en_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    },  
    'virginie_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/es/virginie_riva_subjective_es_text_female_fr_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    }, 
    'samy_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/es/samy_riva_subjective_es_text_male_fr_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    },  
    'virginie_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingualspeechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/fr/virginie_riva_subjective_fr_text_fr_female_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    }, 
    'samy_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/fr/samy_riva_subjective_fr_text_fr_male_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    },  
    'lindy_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/fr/lindy_riva_subjective_fr_text_en_female_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    }, 
    'rodney_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/fr/rodney_riva_subjective_fr_text_en_male_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    },  
    'rubby_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/fr/rubby_riva_subjective_fr_text_es_female_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    }, 
    'carlos_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/fr/carlos_riva_subjective_fr_text_es_male_speaker_normalized_t5_tts_riva_eval_set_normalized_no_escapes.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    },  
    'emotion_en_lindy_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/en/lindy_emotion_t5tts_riva_eval_set.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'emotion_en_rodney_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/en/rodney_emotion_t5tts_riva_eval_set.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'emotion_en_emma_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/en/emma_emotion_t5tts_riva_eval_set.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'emotion_en_sean_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/en/sean_emotion_t5tts_riva_eval_set.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'emotion_en_megan_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/en/megan_emotion_t5tts_riva_eval_set.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'emotion_en_tom_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/en/tom_emotion_t5tts_riva_eval_set.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'emotion_es_rubby_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/es/sp_emotion_rubby_riva_subjective_es_female_es.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es',        
    }, 
    'emotion_es_carlos_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/es/sp_emotion_carlos_riva_subjective_es_male_es.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es',        
    },  
    'emotion_fr_virigine_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/fr/emotion_virginie_riva_subjective_fr_text_fr_female.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',        
    }, 
    'emotion_fr_samy_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/fr/emotion_samy_riva_subjective_fr_text_fr_male.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',        
    },
     'panagram_rubby_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/english_pangrams_es_Speaker0.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'panagram_carlos_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/english_pangrams_es_Speaker1.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'panagram_virginie_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/english_pangrams_fr_Speaker0.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'panagram_sammy_en_iva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/english_pangrams_fr_Speaker1.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'panagram_rubby_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/spanish_pangrams_es_Speaker0.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    }, 
    'panagram_carlos_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/spanish_pangrams_es_Speaker1.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    },  
    'panagram_lindy_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/spanish_pangrams_Lindy.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    }, 
    'panagram_rodney_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/spanish_pangrams_Rodney.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    },  
    'panagram_virginie_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/spanish_pangrams_fr_Speaker0.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    }, 
    'panagram_samy_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/spanish_pangrams_fr_Speaker1.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    },  
    'panagram_virginie_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/french_pangrams_fr_Speaker0.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    }, 
    'panagram_samy_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/french_pangrams_fr_Speaker1.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    },  
    'panagram_lindy_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/french_pangrams_Lindy.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    }, 
    'panagram_rodney_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/french_pangrams_Rodney.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    },  
    'panagram_rubby_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/french_pangrams_es_Speaker0.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    }, 
    'panagram_carlos_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/subjective_riva_manifests/panagram/pangram_json_files/french_pangrams_es_Speaker1.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    },  
    'aed_lindy_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/en/manifest__Language_en_Dataset_Riva_Speaker_Lindy_WIZWIKI.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'aed_rodney_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/en/manifest__Language_en_Dataset_Riva_Speaker_Rodney_WIZWIKI.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'aed_rubby_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/en/manifest_Language_es_Dataset_Riva_es_Speaker_0_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'aed_carlos_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/en/manifest_Language_es_Dataset_Riva_es_Speaker_1_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'aed_virginie_en_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/en/manifest_Language_fr_Dataset_Riva_fr_Speaker_0_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    }, 
    'aed_sammy_en_iva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/en/manifest_Language_fr_Dataset_Riva_fr_Speaker_1_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
    },  
    'aed_rubby_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/es/manifest_Language_es_Dataset_Riva_es_Speaker_0_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    }, 
    'aed_carlos_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/es/manifest_Language_es_Dataset_Riva_es_Speaker_1_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    },  
    'aed_lindy_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/es/manifest__Language_en_Dataset_Riva_Speaker_Lindy_WIZWIKI.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    }, 
    'aed_rodney_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/es/manifest__Language_en_Dataset_Riva_Speaker_Rodney_WIZWIKI.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    },  
    'aed_virginie_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/es/manifest_Language_fr_Dataset_Riva_fr_Speaker_0_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    }, 
    'aed_samy_es_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/es/manifest_Language_fr_Dataset_Riva_fr_Speaker_1_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es', 
    },  
    'aed_virginie_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/fr/manifest_Language_fr_Dataset_Riva_fr_Speaker_0_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    }, 
    'aed_samy_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/fr/manifest_Language_fr_Dataset_Riva_fr_Speaker_1_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    },  
    'aed_lindy_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/fr/manifest__Language_en_Dataset_Riva_Speaker_Lindy_WIZWIKI.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    }, 
    'aed_rodney_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/fr/manifest__Language_en_Dataset_Riva_Speaker_Rodney_WIZWIKI.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    },  
    'aed_rubby_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/fr/manifest_Language_es_Dataset_Riva_es_Speaker_0_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    }, 
    'aed_carlos_fr_riva_subjective': {
        'manifest_path' : '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/crosslingual_speakers_AED/fr/manifest_Language_es_Dataset_Riva_es_Speaker_1_Emotion_Regu.json',
        'audio_dir' : "/datap/misc/Data/riva",
        'feature_dir' : '',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
    },  
    
    'Emma_Riva_de_Additional': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/output/Emma_Riva.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },

    # Sean
    'Sean_Riva_de_Additional': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/output/Sean_Riva.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },

    # Tom
    'Tom_Riva_de_Additional': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/output/Tom_Riva.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },

    # Megan
    'Megan_Riva_de_Additional': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/output/Megan_Riva.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },

    # Lindy_WIZWIKI
    'Lindy_WIZWIKI_Riva_de': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/output/Lindy_WIZWIKI_Riva.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },

    # Rodney_WIZWIKI
    'Rodney_WIZWIKI_Riva_de': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/output/Rodney_WIZWIKI_Riva.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },

    # Riva_es Speaker:0
    'Speaker0_Riva_es_de_Regular': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/output/0_Riva_es.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },

    # Riva_es Speaker:1
    'Speaker1_Riva_es_de_Regular': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/output/1_Riva_es.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },

    # Riva_fr Speaker:0
    'Speaker0_Riva_fr_de_Regular': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/output/0_Riva_fr.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },

    # Riva_fr Speaker:1
    'Speaker1_Riva_fr_de_Regular': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/output/1_Riva_fr.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },
    # Riva_fr Speaker:1
    'de_subjective_emma_Regular': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/de_output/Emma_Riva.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
    },
    # Riva_fr Speaker:1
    'de_subjective_riva_fr_1_Regular': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/de_output/1_Riva_fr.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
    },
     # Riva_fr Speaker:1
    'de_subjective_emma_Regular_char': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/de_output/Emma_Riva.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_chartokenizer'],
        'whisper_language': 'de',
    },
    # Riva_fr Speaker:1
    'de_subjective_riva_fr_1_Regular_char': {
        'manifest_path': '/datap/misc/speechllm_codecdatasets_multilingual/manifests/new_codebase_manifests/german/de_output/1_Riva_fr.json',
        'audio_dir': '/datap/misc/Data/riva',
        'feature_dir': '',
        'tokenizer_names': ['german_chartokenizer'],
        'whisper_language': 'de',
    },
}
