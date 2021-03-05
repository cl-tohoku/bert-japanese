import logging
import unicodedata

from transformers.tokenization_bert_japanese import BertJapaneseTokenizer, CharacterTokenizer


logger = logging.getLogger(__name__)


class BertJapaneseTokenizerForPretraining(BertJapaneseTokenizer):
    """BERT tokenizer for Japanese text"""

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        do_word_tokenize=True,
        do_subword_tokenize=True,
        word_tokenizer_type="basic",
        subword_tokenizer_type="wordpiece",
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        mecab_kwargs=None,
        manual_subword_marking=False,
        **kwargs
    ):
        """Constructs a MecabBertTokenizer.
        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file.
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input.
                Only has an effect when do_basic_tokenize=True.
            **do_word_tokenize**: (`optional`) boolean (default True)
                Whether to do word tokenization.
            **do_subword_tokenize**: (`optional`) boolean (default True)
                Whether to do subword tokenization.
            **word_tokenizer_type**: (`optional`) string (default "basic")
                Type of word tokenizer.
            **subword_tokenizer_type**: (`optional`) string (default "wordpiece")
                Type of subword tokenizer.
            **mecab_kwargs**: (`optional`) dict passed to `MecabTokenizer` constructor (default None)
            **manual_subword_marking**: (`optional`) bool (default None)
                Whether the subword markers are appended manually.
        """
        super().__init__(
            vocab_file,
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
            **kwargs
        )

        self.manual_subword_marking = manual_subword_marking
        if do_subword_tokenize and subword_tokenizer_type == "character":
            self.subword_tokenizer = CharacterTokenizerForPretraining(
                vocab=self.vocab, unk_token=self.unk_token, add_subword_markers=self.manual_subword_marking
            )

    def _tokenize(self, text):
        if self.do_word_tokenize:
            tokens = self.word_tokenizer.tokenize(text, never_split=self.all_special_tokens)
        else:
            tokens = [text]

        if self.do_subword_tokenize:
            split_tokens = [sub_token for token in tokens for sub_token in self.subword_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens

        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if self.manual_subword_marking and token[:2] == "##":
            return self.vocab.get(token[2:], self.vocab.get(self.unk_token))
        else:
            return self.vocab.get(token, self.vocab.get(self.unk_token))


class CharacterTokenizerForPretraining(CharacterTokenizer):
    """Runs Character tokenziation."""

    def __init__(self, vocab, unk_token, normalize_text=True, add_subword_markers=False):
        """Constructs a CharacterTokenizer.
        Args:
            **vocab**:
                Vocabulary object.
            **unk_token**: str
                A special symbol for out-of-vocabulary token.
            **normalize_text**: (`optional`) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
            **add_subword_marker**: (optional`) boolean (default False)
                If set to True, the subword marker "##" will be prepended to i-th (i > 0) characters within each word.
        """
        super().__init__(vocab, unk_token, normalize_text=normalize_text)
        self.add_subword_markers = add_subword_markers

    def tokenize(self, text):
        """Tokenizes a piece of text into characters.
        For example:
            input = "apple"
            output = ["a", "##p", "##p", "##l", "##e"]  (if add_subword_markers is True)
                     ["a", "p", "p", "l", "e"]          (if add_subword_markers is False)
        Args:
            text: A single token or whitespace separated tokens.
                This should have already been passed through `BasicTokenizer`.
        Returns:
            A list of characters.
        """
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        output_tokens = []
        for i, char in enumerate(text):
            if char not in self.vocab:
                output_tokens.append(self.unk_token)
                continue

            if self.add_subword_markers and i > 0:
                output_tokens.append("##{}".format(char))
            else:
                output_tokens.append(char)

        return output_tokens
