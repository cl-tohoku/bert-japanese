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
        self._tokenizer.add_special_tokens(['<unused{}>'.format(i) for i in range(num_unused_tokens)])

        self._tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Strip()])
        if mecab_dic_type in ("unidic_lite", "unidic", "ipadic"):
            self._tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(MeCabPreTokenizer(mecab_dic_type))
        elif mecab_dic_type == "whitespace":
            self._tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        else:
            raise ValueError("Invalid pre_tokenizer_type is specified.")

        parameters = {
            "model": "BertWordPieceJapaneseTokenizer",
            "mecab_dic_type": mecab_dic_type,
        }
        self._parameters.update(parameters)
