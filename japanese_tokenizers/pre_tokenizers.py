import os
from typing import List, Optional, Tuple

from tokenizers import NormalizedString, PreTokenizedString

Offsets = Tuple[int, int]


class MeCabPreTokenizer:
    def __init__(self,
                 mecab_dic: Optional[str] = None,
                 mecab_option: Optional[str] = None) -> None:
        import fugashi
        mecab_option = mecab_option or ""

        if mecab_dic is not None:
            if mecab_dic == "unidic_lite":
                import unidic_lite
                dic_dir = unidic_lite.DICDIR
            elif mecab_dic == "unidic":
                import unidic
                dic_dir = unidic.DICDIR
            elif mecab_dic == "ipadic":
                import ipadic
                dic_dir = ipadic.DICDIR
            else:
                raise ValueError("Invalid mecab_dic is specified.")

            mecabrc = os.path.join(dic_dir, "mecabrc")
            mecab_option = "-d {} -r {} ".format(dic_dir, mecabrc) + mecab_option

        self.mecab = fugashi.GenericTagger(mecab_option)

    def mecab_split(self, i: int, text: NormalizedString) -> List[NormalizedString]:
        tokens = []
        cursor = 0
        for word in self.mecab(str(text)):
            token = word.surface
            start = str(text).index(token, cursor)
            end = start + len(token)

            tokens.append(text[start:end])
            cursor = end

        return tokens

    def pre_tokenize(self, text: PreTokenizedString):
        text.split(self.mecab_split)
