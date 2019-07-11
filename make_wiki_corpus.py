import os
import re
import json
import gzip
import argparse

import MeCab
import tensorflow as tf


N_DOCS_DEBUG = 1000
N_DOCS_PER_FILE = 100000


class MeCabSentenceSplitter(object):
    def __init__(self, dict_path=None):
        if dict_path is not None:
            self.mecab_tagger = MeCab.Tagger(f'-d {dict_path}')
        else:
            self.mecab_tagger = MeCab.Tagger()

    def __call__(self, text):
        sentences = []
        start = 0
        end = 0
        for line in self.mecab_tagger.parse(text).split('\n'):
            if line == 'EOS':
                break

            token, morph_info = line.split('\t')
            end = text.index(token, end) + len(token)
            if morph_info.startswith('記号,句点,'):
                sentences.append(text[start:end])
                start = end

        return sentences


def preprocess_text(text, title=None):
    # remove navigation preceding body text
    if title:
        nav_segment = f'> {title}'
        while nav_segment in text:
            nav_end = text.index(nav_segment) + len(nav_segment)
            text = text[nav_end:]

    # squeeze repeated whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def postprocess_text(text):
    # remove text beginning with a caret
    text = re.sub(r'\^.*', '', text)
    return text.strip()


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    sent_splitter = MeCabSentenceSplitter(args.mecab_dict)

    tf.gfile.MakeDirs(args.output_dir)

    n_processed_docs = 0
    with tf.gfile.GFile(args.cirrus_file, 'rb') as reader, \
         gzip.GzipFile(fileobj=reader) as cirrus_file:
        for line in cirrus_file:
            json_obj = json.loads(line)
            # ignore non-article pages
            if json_obj.get('namespace', -1) != 0:
                continue
            # ignore articles with few incoming links
            if json_obj.get('incoming_links', 0) < args.min_inlinks:
                continue
            # ignore disambiguation pages
            templates = json_obj.get('template') or []
            if 'Template:Dmbox' in templates:
                continue

            title = json_obj['title']
            text = json_obj.get('text') or ''
            text = preprocess_text(text, title)
            sentences = [postprocess_text(sent) for sent in sent_splitter(text)]
            # ignore too short/long sentences
            sentences = [sent for sent in sentences
                         if args.min_length <= len(sent) <= args.max_length]

            if sentences:
                # open a new file at certain intervals
                if n_processed_docs % N_DOCS_PER_FILE == 0:
                    file_id = n_processed_docs // N_DOCS_PER_FILE
                    output_file_path = \
                        os.path.join(args.output_dir, f'corpus_{file_id:03d}.txt')
                    output_file = tf.gfile.GFile(output_file_path, 'w')
                    tf.logging.info(f'writing to {output_file_path}')

                # write document to a file
                for sent in sentences:
                    assert not '\n' in sent, sent
                    assert sent, sent
                    output_file.write(f'{sent}\n')

                output_file.write('\n')

                n_processed_docs += 1
                if args.debug and n_processed_docs == N_DOCS_DEBUG:
                    tf.logging.info(f'processed: {n_processed_docs}')
                    break

                # logging
                if n_processed_docs % (N_DOCS_PER_FILE // 10) == 0:
                    tf.logging.info(f'processed: {n_processed_docs}')

    if not args.debug and n_processed_docs % N_DOCS_PER_FILE != 0:
        tf.logging.info(f'processed: {n_processed_docs}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cirrus_file', type=str,
        help='Wikipedia Cirrussearch dump file (.json.gz)')
    parser.add_argument('output_dir', type=str,
        help='output directory to write corpus files')
    parser.add_argument('--min_inlinks', type=int, default=1,
        help='limit document size by number of incoming links [1]')
    parser.add_argument('--min_length', type=int, default=20,
        help='only extract sentences with N+ characters [20]')
    parser.add_argument('--max_length', type=int, default=1000,
        help='only extract sentences with N+ characters [1000]')
    parser.add_argument('--mecab_dict', type=str,
        help='path to MeCab dictionary')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
