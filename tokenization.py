# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright 2023 Masatoshi Suzuki (@singletongue)
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

import unicodedata

from transformers.models.bert_japanese.tokenization_bert_japanese import (
    BertJapaneseTokenizer as BertJapaneseTokenizerBase,
    CharacterTokenizer as CharacterTokenizerBase,
)


class BertJapaneseTokenizer(BertJapaneseTokenizerBase):
    def __init__(
        self,
        vocab_file,
        spm_file=None,
        do_lower_case=False,
        do_word_tokenize=True,
        do_subword_tokenize=True,
        word_tokenizer_type="basic",
        subword_tokenizer_type="wordpiece",
        vocab_has_no_subword_prefix=False,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        mecab_kwargs=None,
        sudachi_kwargs=None,
        jumanpp_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            spm_file=spm_file,
            do_lower_case=do_lower_case,
            do_word_tokenize=do_word_tokenize,
            do_subword_tokenize=do_subword_tokenize,
            word_tokenizer_type=word_tokenizer_type,
            subword_tokenizer_type=subword_tokenizer_type,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            mecab_kwargs=mecab_kwargs,
            sudachi_kwargs=sudachi_kwargs,
            jumanpp_kwargs=jumanpp_kwargs,
            **kwargs,
        )

        self.vocab_has_no_subword_prefix = vocab_has_no_subword_prefix

        if do_subword_tokenize and subword_tokenizer_type == "character":
            self.subword_tokenizer = CharacterTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    def _convert_token_to_id(self, token):
        if self.vocab_has_no_subword_prefix and token.startswith("##"):
            token = token[len("##"):]

        return self.vocab.get(token, self.vocab.get(self.unk_token))


class CharacterTokenizer(CharacterTokenizerBase):
    def __init__(self, vocab, unk_token, normalize_text=True):
        super().__init__(vocab, unk_token, normalize_text=normalize_text)

    def tokenize(self, text):
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        output_tokens = []
        for i, char in enumerate(text):
            if char not in self.vocab:
                output_tokens.append(self.unk_token)
                continue

            if i > 0:
                char = "##" + char

            output_tokens.append(char)

        return output_tokens
