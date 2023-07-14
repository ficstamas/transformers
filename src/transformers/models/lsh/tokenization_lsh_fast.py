# coding=utf-8
# Copyright 2023 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Fast Tokenization classes for LSH."""

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_lsh import LSHTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "ficsort/lsh-base": "https://huggingface.co/ficsort/lsh-base/resolve/main/vocab.txt",
        "lsh-large-uncased": "https://huggingface.co/lsh-large-uncased/resolve/main/vocab.txt",
        "lsh-base-cased": "https://huggingface.co/lsh-base-cased/resolve/main/vocab.txt",
        "lsh-large-cased": "https://huggingface.co/lsh-large-cased/resolve/main/vocab.txt",
        "lsh-base-multilingual-uncased": (
            "https://huggingface.co/lsh-base-multilingual-uncased/resolve/main/vocab.txt"
        ),
        "lsh-base-multilingual-cased": "https://huggingface.co/lsh-base-multilingual-cased/resolve/main/vocab.txt",
        "lsh-base-chinese": "https://huggingface.co/lsh-base-chinese/resolve/main/vocab.txt",
        "lsh-base-german-cased": "https://huggingface.co/lsh-base-german-cased/resolve/main/vocab.txt",
        "lsh-large-uncased-whole-word-masking": (
            "https://huggingface.co/lsh-large-uncased-whole-word-masking/resolve/main/vocab.txt"
        ),
        "lsh-large-cased-whole-word-masking": (
            "https://huggingface.co/lsh-large-cased-whole-word-masking/resolve/main/vocab.txt"
        ),
        "lsh-large-uncased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/lsh-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
        ),
        "lsh-large-cased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/lsh-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
        ),
        "lsh-base-cased-finetuned-mrpc": (
            "https://huggingface.co/lsh-base-cased-finetuned-mrpc/resolve/main/vocab.txt"
        ),
        "lsh-base-german-dbmdz-cased": "https://huggingface.co/lsh-base-german-dbmdz-cased/resolve/main/vocab.txt",
        "lsh-base-german-dbmdz-uncased": (
            "https://huggingface.co/lsh-base-german-dbmdz-uncased/resolve/main/vocab.txt"
        ),
        "TurkuNLP/lsh-base-finnish-cased-v1": (
            "https://huggingface.co/TurkuNLP/lsh-base-finnish-cased-v1/resolve/main/vocab.txt"
        ),
        "TurkuNLP/lsh-base-finnish-uncased-v1": (
            "https://huggingface.co/TurkuNLP/lsh-base-finnish-uncased-v1/resolve/main/vocab.txt"
        ),
        "wietsedv/lsh-base-dutch-cased": (
            "https://huggingface.co/wietsedv/lsh-base-dutch-cased/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "ficsort/lsh-base": "https://huggingface.co/ficsort/lsh-base/resolve/main/tokenizer.json",
        "lsh-large-uncased": "https://huggingface.co/lsh-large-uncased/resolve/main/tokenizer.json",
        "lsh-base-cased": "https://huggingface.co/lsh-base-cased/resolve/main/tokenizer.json",
        "lsh-large-cased": "https://huggingface.co/lsh-large-cased/resolve/main/tokenizer.json",
        "lsh-base-multilingual-uncased": (
            "https://huggingface.co/lsh-base-multilingual-uncased/resolve/main/tokenizer.json"
        ),
        "lsh-base-multilingual-cased": (
            "https://huggingface.co/lsh-base-multilingual-cased/resolve/main/tokenizer.json"
        ),
        "lsh-base-chinese": "https://huggingface.co/lsh-base-chinese/resolve/main/tokenizer.json",
        "lsh-base-german-cased": "https://huggingface.co/lsh-base-german-cased/resolve/main/tokenizer.json",
        "lsh-large-uncased-whole-word-masking": (
            "https://huggingface.co/lsh-large-uncased-whole-word-masking/resolve/main/tokenizer.json"
        ),
        "lsh-large-cased-whole-word-masking": (
            "https://huggingface.co/lsh-large-cased-whole-word-masking/resolve/main/tokenizer.json"
        ),
        "lsh-large-uncased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/lsh-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json"
        ),
        "lsh-large-cased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/lsh-large-cased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json"
        ),
        "lsh-base-cased-finetuned-mrpc": (
            "https://huggingface.co/lsh-base-cased-finetuned-mrpc/resolve/main/tokenizer.json"
        ),
        "lsh-base-german-dbmdz-cased": (
            "https://huggingface.co/lsh-base-german-dbmdz-cased/resolve/main/tokenizer.json"
        ),
        "lsh-base-german-dbmdz-uncased": (
            "https://huggingface.co/lsh-base-german-dbmdz-uncased/resolve/main/tokenizer.json"
        ),
        "TurkuNLP/lsh-base-finnish-cased-v1": (
            "https://huggingface.co/TurkuNLP/lsh-base-finnish-cased-v1/resolve/main/tokenizer.json"
        ),
        "TurkuNLP/lsh-base-finnish-uncased-v1": (
            "https://huggingface.co/TurkuNLP/lsh-base-finnish-uncased-v1/resolve/main/tokenizer.json"
        ),
        "wietsedv/lsh-base-dutch-cased": (
            "https://huggingface.co/wietsedv/lsh-base-dutch-cased/resolve/main/tokenizer.json"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "ficsort/lsh-base": 512,
    "lsh-large-uncased": 512,
    "lsh-base-cased": 512,
    "lsh-large-cased": 512,
    "lsh-base-multilingual-uncased": 512,
    "lsh-base-multilingual-cased": 512,
    "lsh-base-chinese": 512,
    "lsh-base-german-cased": 512,
    "lsh-large-uncased-whole-word-masking": 512,
    "lsh-large-cased-whole-word-masking": 512,
    "lsh-large-uncased-whole-word-masking-finetuned-squad": 512,
    "lsh-large-cased-whole-word-masking-finetuned-squad": 512,
    "lsh-base-cased-finetuned-mrpc": 512,
    "lsh-base-german-dbmdz-cased": 512,
    "lsh-base-german-dbmdz-uncased": 512,
    "TurkuNLP/lsh-base-finnish-cased-v1": 512,
    "TurkuNLP/lsh-base-finnish-uncased-v1": 512,
    "wietsedv/lsh-base-dutch-cased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "ficsort/lsh-base": {"do_lower_case": True},
    "lsh-large-uncased": {"do_lower_case": True},
    "lsh-base-cased": {"do_lower_case": False},
    "lsh-large-cased": {"do_lower_case": False},
    "lsh-base-multilingual-uncased": {"do_lower_case": True},
    "lsh-base-multilingual-cased": {"do_lower_case": False},
    "lsh-base-chinese": {"do_lower_case": False},
    "lsh-base-german-cased": {"do_lower_case": False},
    "lsh-large-uncased-whole-word-masking": {"do_lower_case": True},
    "lsh-large-cased-whole-word-masking": {"do_lower_case": False},
    "lsh-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
    "lsh-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
    "lsh-base-cased-finetuned-mrpc": {"do_lower_case": False},
    "lsh-base-german-dbmdz-cased": {"do_lower_case": False},
    "lsh-base-german-dbmdz-uncased": {"do_lower_case": True},
    "TurkuNLP/lsh-base-finnish-cased-v1": {"do_lower_case": False},
    "TurkuNLP/lsh-base-finnish-uncased-v1": {"do_lower_case": True},
    "wietsedv/lsh-base-dutch-cased": {"do_lower_case": False},
}


class LSHTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" LSH tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original LSH).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = LSHTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A LSH sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A LSH sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
