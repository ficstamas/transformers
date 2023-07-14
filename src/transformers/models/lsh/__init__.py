# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tensorflow_text_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_lsh": ["LSH_PRETRAINED_CONFIG_ARCHIVE_MAP", "LSHConfig", "LSHOnnxConfig"],
    "tokenization_lsh": ["BasicTokenizer", "LSHTokenizer", "WordpieceTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_lsh_fast"] = ["LSHTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_lsh"] = [
        "LSH_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LSHForMaskedLM",
        "LSHForMultipleChoice",
        "LSHForNextSentencePrediction",
        "LSHForPreTraining",
        "LSHForQuestionAnswering",
        "LSHForSequenceClassification",
        "LSHForTokenClassification",
        "LSHLayer",
        "LSHLMHeadModel",
        "LSHModel",
        "LSHPreTrainedModel",
        "load_tf_weights_in_lsh",
    ]

try:
    if not is_tensorflow_text_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_lsh_tf"] = ["TFLSHTokenizer"]

if TYPE_CHECKING:
    from .configuration_lsh import LSH_PRETRAINED_CONFIG_ARCHIVE_MAP, LSHConfig, LSHOnnxConfig
    from .tokenization_lsh import BasicTokenizer, LSHTokenizer, WordpieceTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_lsh_fast import LSHTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_lsh import (
            LSH_PRETRAINED_MODEL_ARCHIVE_LIST,
            LSHForMaskedLM,
            LSHForMultipleChoice,
            LSHForNextSentencePrediction,
            LSHForPreTraining,
            LSHForQuestionAnswering,
            LSHForSequenceClassification,
            LSHForTokenClassification,
            LSHLayer,
            LSHLMHeadModel,
            LSHModel,
            LSHPreTrainedModel,
            load_tf_weights_in_lsh,
        )

    try:
        if not is_tensorflow_text_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_lsh_tf import TFLSHTokenizer

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
