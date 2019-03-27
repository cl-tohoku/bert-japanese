# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes."""

import collections
import unicodedata

import MeCab
import tensorflow as tf


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = reader.readline()
            if not token:
                break

            token = token.strip()
            vocab[token] = index
            index += 1

    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class JapaneseBertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file,
                 vocab_type="bpe", do_lower_case=True, mecab_dict=None):
        """Constructs a BertTokenizer.

        Args:
            vocab_file: Path to a one-wordpiece-per-line vocabulary file.
            vocab_type: Subword vocabulary type ("bpe" or "char").
            do_lower_case: Whether to lower case the input.
            mecab_dict: Path to MeCab custom dictionary.
        """

        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = JapaneseBasicTokenizer(do_lower_case, mecab_dict)
        if vocab_type == "bpe":
            self.subword_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        elif vocab_type == "char":
            self.subword_tokenizer = CharacterTokenizer(vocab=self.vocab)
        else:
            raise RuntimeError(f"Invalid vocab_type {vocab_type} is specified.")

    def tokenize(self, text, with_flags=False):
        split_tokens = []
        for tokenized_text in self.basic_tokenizer.tokenize(text):
            for sub_token in self.subword_tokenizer.tokenize(tokenized_text, with_flags):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of wordpiece tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])

        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of wordpiece ids into tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.inv_vocab[i])

        return tokens


class JapaneseBasicTokenizer(object):
    """Runs basic tokenization (such as lower casing) and Japanese tokenization."""

    def __init__(self, do_lower_case=True, mecab_dict=None):
        """Constructs a BasicTokenizer.

        Args:
            do_lower_case: Whether to lower case the input.
            mecab_dict: Path to MeCab custom dictionary.
        """
        self.do_lower_case = do_lower_case
        if mecab_dict is not None:
            self.mecab_tagger = MeCab.Tagger(f"-d {mecab_dict} -O wakati")
        else:
            self.mecab_tagger = MeCab.Tagger("-O wakati")

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        text = self._normalize_text(text)
        text = self._tokenize_japanese_words(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()

            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _normalize_text(self, text):
        """Apply NFKC normalization to a text."""
        text = unicodedata.normalize("NFKC", text)
        return text

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_japanese_words(self, text):
        """Split Japanese text into tokens with whitespaces using MeCab."""
        tokenized_text = self.mecab_tagger.parse(text)
        tokenized_text = tokenized_text.strip()
        return tokenized_text

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text, with_flags=False):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `JapaneseBasicTokenizer`.
            with_flags: If set to True, flags indicating whether each token in the
                beginnging of a word is returned, as well as list of tokens.

        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                if with_flags:
                    output_tokens.append((self.unk_token, 1))
                else:
                    output_tokens.append(self.unk_token)

                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr

                    if substr in self.vocab:
                        cur_substr = substr
                        break

                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                if with_flags:
                    sub_tokens.append((cur_substr, int(start == 0)))
                else:
                    sub_tokens.append(cur_substr)

                start = end

            if is_bad:
                if with_flags:
                    output_tokens.append((self.unk_token, 1))
                else:
                    output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


class CharacterTokenizer(object):
    """Runs Character tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text, with_flags=False):
        """Tokenizes a piece of text into its characters.

        For example:
            input = "東北 大学"
            output = ["東", "北", "大", "学"]

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `JapaneseBasicTokenizer`.
            with_flags: If set to True, flags indicating whether each token in the
                beginnging of a word is returned, as well as list of tokens.

        Returns:
            A list of character tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                if with_flags:
                    output_tokens.append((self.unk_token, 1))
                else:
                    output_tokens.append(self.unk_token)

                continue

            for i, char in enumerate(chars):
                if char in self.vocab:
                    if with_flags:
                        output_tokens.append((char, int(i == 0)))
                    else:
                        output_tokens.append(char)
                else:
                    if with_flags:
                        output_tokens.append((self.unk_token, int(i == 0)))
                    else:
                        output_tokens.append(self.unk_token)

        return output_tokens


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True

    cat = unicodedata.category(char)
    if cat == "Zs":
        return True

    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False

    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True

    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True

    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True

    return False
