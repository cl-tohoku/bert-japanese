# Copyright 2020 The HuggingFace Inc. team.
# Copyright 2023 Masatoshi Suzuki (@singletongue).
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

from typing import Dict, Optional, Union

from tokenizers import AddedToken, normalizers, pre_tokenizers
from tokenizers.implementations import BertWordPieceTokenizer

from .pre_tokenizers import MeCabPreTokenizer


class JapaneseWordPieceTokenizer(BertWordPieceTokenizer):
    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        unk_token: Union[str, AddedToken] = "[UNK]",
        sep_token: Union[str, AddedToken] = "[SEP]",
        cls_token: Union[str, AddedToken] = "[CLS]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        mask_token: Union[str, AddedToken] = "[MASK]",
        num_unused_tokens: int = 10,
        pre_tokenizer_type: str = "mecab",
        mecab_dic_type: str = "unidic_lite",
        wordpieces_prefix: str = "##",
    ) -> None:
        super().__init__(
            vocab=vocab,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            wordpieces_prefix=wordpieces_prefix,
        )
        self._tokenizer.add_special_tokens([f"[unused{i}]" for i in range(num_unused_tokens)])

        self._tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Strip()])

        parameters = {
            "model": "BertWordPieceJapaneseTokenizer",
            "pre_tokenizer_type": pre_tokenizer_type,
            "mecab_dic_type": mecab_dic_type,
        }

        if pre_tokenizer_type == "mecab":
            self._tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(MeCabPreTokenizer(mecab_dic_type))
            parameters["mecab_dic_type"] = mecab_dic_type
        elif pre_tokenizer_type == "whitespace":
            self._tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        else:
            raise ValueError("Invalid pre_tokenizer_type is specified.")

        self._parameters.update(parameters)
