import argparse
import gzip
import json
import os
import re
import unicodedata

from tqdm import tqdm


class MeCabSentenceSplitter(object):
    def __init__(self, mecab_option=None):
        import fugashi
        if mecab_option is None:
            import unidic_lite
            dic_dir = unidic_lite.DICDIR
            mecabrc = os.path.join(dic_dir, "mecabrc")
            mecab_option = "-d {} -r {}".format(dic_dir, mecabrc)

        self.mecab = fugashi.GenericTagger(mecab_option)

    def __call__(self, text):
        sentences = []
        start = 0
        end = 0
        for line in self.mecab.parse(text).split("\n"):
            if line == "EOS":
                if len(text[start:]) > 0:
                    sentences.append(text[start:])
                break

            token, token_info = line.split("\t", maxsplit=1)
            end = text.index(token, end) + len(token)
            if "記号" in token_info and "句点" in token_info:
                sentences.append(text[start:end])
                start = end

        return sentences


def preprocess_text(text, title=None):
    text = unicodedata.normalize("NFKC", text)

    # remove invisible characters
    text = "".join(c for c in text if c.isprintable())

    # remove templates
    text = re.sub(r"\[\d+?\]", "", text)
    text = re.sub(r"\[要.+?\]", "", text)
    text = re.sub(r"\{\{+[^{}]+?\}\}+", "", text)

    # remove navigation
    if title is not None:
        text = re.sub(r"^.+? \> " + re.escape(title), "", text)

    # remove footnotes
    text = re.sub(r" \^ .+", "", text)
    # remove annotations
    text = re.sub(r"\[(要出典|リンク切れ|.+?\?)\]", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_text(text):
    # filter out text containing equations
    if "\displaystyle" in text:
        return False

    return True


def main(args):
    sent_splitter = MeCabSentenceSplitter(args.mecab_option)

    with gzip.open(args.input_file, "rt") as input_file, \
         open(args.output_file, "w") as output_file:
        for line in tqdm(input_file):
            json_item = json.loads(line)
            text = json_item.get("text")
            if text is None:
                continue

            title = json_item.get("title")
            text = preprocess_text(text, title=title)

            is_processed = False
            for sentence in sent_splitter(text):
                sentence = sentence.strip()
                if len(sentence) < args.min_text_length:
                    continue
                if len(sentence) > args.max_text_length:
                    continue
                if not filter_text(sentence):
                    continue

                assert not "\n" in text
                assert sentence != ""
                print(sentence, file=output_file)
                is_processed = True

            if is_processed:
                print("", file=output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--min_text_length", type=int, default=10)
    parser.add_argument("--max_text_length", type=int, default=1000)
    parser.add_argument("--mecab_option", type=str)
    args = parser.parse_args()
    main(args)
