import tempfile
import os
import argparse
from collections import Counter

import sentencepiece
import tensorflow as tf
from tensor2tensor.data_generators import text_encoder

from tokenization import MecabBasicTokenizer


CONTROL_SYMBOLS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    token_counts = Counter()
    tokenizer = MecabBasicTokenizer(do_lower_case=args.do_lower_case,
                                    mecab_dict_path=args.mecab_dict_path)

    for input_file in tf.gfile.Glob(args.input_file):
        with tf.gfile.GFile(input_file, 'r') as reader:
            tf.logging.info('reading {}'.format(input_file))
            for line in reader:
                tokens, _ = tokenizer.tokenize(line.strip('\n'))
                token_counts.update(tokens)

    tf.logging.info('building WordPiece text encoder')
    bpe_encoder = text_encoder.SubwordTextEncoder()
    bpe_encoder.build_from_token_counts(token_counts, args.min_count)

    with tf.gfile.GFile(args.output_file, 'w') as output_file:
        tf.logging.info('writing vocabulary to {}'.format(output_file))

        for symbol in CONTROL_SYMBOLS:
            output_file.write(symbol + '\n')

        for subtoken in bpe_encoder.all_subtoken_strings:
            # ignore pre-defined WordPiece symbols
            if subtoken in ('<pad>_', '<EOS>_'):
                continue

            if subtoken.endswith('_'):
                # e.g. "word_" -> "word"
                output_token = subtoken[:-1]
            else:
                # e.g. "tion" -> "##tion"
                output_token = '##' + subtoken

            output_file.write(output_token + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
        help='Input raw text file (or comma-separated list of files).')
    parser.add_argument('--output_file', type=str, required=True,
        help='Output vocabulary file.')
    parser.add_argument('--min_count', type=int, default=5,
        help='Minimum count of subtokens in corpus. [5]')
    parser.add_argument('--do_lower_case', type=bool, default=False,
        help='Whether to lower case the input text. [False]')
    parser.add_argument('--mecab_dict_path', type=str,
        help='Path to MeCab custom dictionary.')
    args = parser.parse_args()

    main(args)
