# BERT Models for Japanese Text

BERT models trained on Japanese text.

## Features

- All the models are trained on Japanese Wikipedia.
- We trained models with different tokenization algorithms.
    - **`mecab-ipadic-bpe-32k`**: texts are first tokenized with [MeCab](https://taku910.github.io/mecab) morphological parser and then split into subwords by WordPiece. The vocabulary size is 32000.
    - **`mecab-ipadic-char-4k`**: texts are first tokenized with MeCab and then split into characters (information of MeCab tokenization is preserved). The vocabulary size is 4000.
- All the models are trained with the same configuration as the original BERT; 512 tokens per instance, 256 instances per batch, and 1M training steps.
- We also distribute models trained with **Whole Word Masking** enabled; all of the tokens corresponding to a word (tokenized by MeCab) are masked at once.
- Along with the models, we provide [tokenizers](https://github.com/singletongue/japanese-bert/blob/master/tokenization.py), which are compatible with ones defined in [Transformers](https://github.com/huggingface/transformers) by Hugging Face.

## Pretrained models

- BERT-base models (12-layer, 768-hidden, 12-heads, 110M parameters)
    - **[`BERT-base_mecab-ipadic-bpe-32k.tar.xz`](https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-bpe-32k.tar.xz)** (2.1GB)
        - MeCab + WordPiece tokenization.
    - **[`BERT-base_mecab-ipadic-bpe-32k_whole-word-mask.tar.xz`](https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-bpe-32k_whole-word-mask.tar.xz)** (2.1GB)
        - MeCab + WordPiece tokenization. Whole Word Masking is enabled during training.
    - **[`BERT-base_mecab-ipadic-char-4k.tar.xz`](https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-char-4k.tar.xz)** (1.6GB)
        - Character tokenization.
    - **[`BERT-base_mecab-ipadic-char-4k_whole-word-mask.tar.xz`](https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-char-4k_whole-word-mask.tar.xz)** (1.6GB)
        - Character tokenization. Whole Word Masking is enabled during training (word boundaries are determined by MeCab).

All the model archives include following files.
`pytorch_model.bin` and `tf_model.h5` are compatible with [Transformers](https://github.com/huggingface/transformers).

```
.
├── config.json
├── model.ckpt.data-00000-of-00001
├── model.ckpt.index
├── model.ckpt.meta
├── pytorch_model.bin
├── tf_model.h5
└── vocab.txt
```

At present, only `BERT-base` models are available.
We are planning to release `BERT-large` models in the future.

## Usage

Refer to [`masked_lm_example.ipynb`](https://github.com/singletongue/japanese-bert/blob/master/masked_lm_example.ipynb).

## Requirements

For just using the models with [`tokenizers.py`](https://github.com/singletongue/japanese-bert/blob/master/tokenization.py):

- [Transformers](https://github.com/huggingface/transformers) (>= 2.1.1)
- [mecab-python3](https://github.com/SamuraiT/mecab-python3) with [MeCab](https://taku910.github.io/mecab) installed

If you wish to pretrain a model:

- [TensorFlow](https://github.com/tensorflow/tensorflow) (== 1.14.0)
- [SentencePiece](https://github.com/google/sentencepiece)
- [logzero](https://github.com/metachris/logzero)

## Details of pretraining

### Corpus generation and preprocessing

The all distributed models are pretrained on Japanese Wikipedia.
To generate the corpus, [WikiExtractor](https://github.com/attardi/wikiextractor) is used to extract plain texts from a Wikipedia dump file.

```
$ python WikiExtractor.py --output /path/to/corpus/dir --bytes 512M --compress --json --links --namespaces 0 --no_templates --min_text_length 16 --processes 20 jawiki-20190901-pages-articles-multistream.xml.bz2
```

Some preprocessing is applied to the extracted texts.
Preprocessing includes splitting texts into sentences, removing noisy markups, etc.

Here we used [mecab-ipadic-NEologd](https://github.com/neologd/mecab-ipadic-neologd) to handle proper nouns correctly (i.e. not to treat `。` in named entities such as `モーニング娘。` and `ゲスの極み乙女。` as sentence boundaries.)

```sh
$ seq -f %02g 0 8|xargs -L 1 -I {} -P 9 python make_corpus.py --input_file /path/to/corpus/dir/AA/wiki_{}.bz2 --output_file /path/to/corpus/dir/corpus.txt.{} --mecab_dict_path /path/to/neologd/dict/dir/
```

### Building vocabulary

Same as the original BERT, we used byte-pair-encoding (BPE) to obtain subwords.
We used a implementaion of BPE in [SentencePiece](https://github.com/google/sentencepiece).

```sh
# For mecab-ipadic-bpe-32k models
$ python build_vocab.py --input_file "/path/to/corpus/dir/corpus.txt.*" --output_file "/path/to/base/dir/vocab.txt" --subword_type bpe --vocab_size 32000

# For mecab-ipadic-char-4k models
$ python build_vocab.py --input_file "/path/to/corpus/dir/corpus.txt.*" --output_file "/path/to/base/dir/vocab.txt" --subword_type char --vocab_size 4000
```

### Creating data for pretraining

With the vocabulary and text files above, we create dataset files for pretraining.
Note that this process is highly memory-consuming and takes many hours.

```sh
# For mecab-ipadic-bpe-32k w/ whole word masking
# Note: each process will consume about 32GB RAM
$ seq -f %02g 0 8|xargs -L 1 -I {} -P 1 python create_pretraining_data.py --input_file /path/to/corpus/dir/corpus.txt.{} --output_file /path/to/base/dir/pretraining-data.tf_record.{} --do_whole_word_mask True --vocab_file /path/to/base/dir/vocab.txt --subword_type bpe --max_seq_length 512 --max_predictions_per_seq 80 --masked_lm_prob 0.15

# For mecab-ipadic-bpe-32k w/o whole word masking
# Note: each process will consume about 32GB RAM
$ seq -f %02g 0 8|xargs -L 1 -I {} -P 1 python create_pretraining_data.py --input_file /path/to/corpus/dir/corpus.txt.{} --output_file /path/to/base/dir/pretraining-data.tf_record.{} --vocab_file /path/to/base/dir/vocab.txt --subword_type bpe --max_seq_length 512 --max_predictions_per_seq 80 --masked_lm_prob 0.15

# For mecab-ipadic-char-4k w whole word masking
# Note: each process will consume about 45GB RAM
$ seq -f %02g 0 8|xargs -L 1 -I {} -P 1 python create_pretraining_data.py --input_file /path/to/corpus/dir/corpus.txt.{} --output_file /path/to/base/dir/pretraining-data.tf_record.{} --do_whole_word_mask True --vocab_file /path/to/base/dir/vocab.txt --subword_type char --max_seq_length 512 --max_predictions_per_seq 80 --masked_lm_prob 0.15

# For mecab-ipadic-char-4k w/o whole word masking
# Note: each process will consume about 45GB RAM
$ seq -f %02g 0 8|xargs -L 1 -I {} -P 1 python create_pretraining_data.py --input_file /path/to/corpus/dir/corpus.txt.{} --output_file /path/to/base/dir/pretraining-data.tf_record.{} --vocab_file /path/to/base/dir/vocab.txt --subword_type char --max_seq_length 512 --max_predictions_per_seq 80 --masked_lm_prob 0.15
```

### Training

We used [Cloud TPUs](https://cloud.google.com/tpu/) to run pre-training.

For BERT-base models, v3-8 TPUs are used.

```sh
# For mecab-ipadic-bpe-32k BERT-base models
$ python3 run_pretraining.py \
--input_file="/path/to/pretraining-data.tf_record.*" \
--output_dir="/path/to/output_dir" \
--bert_config_file=bert_base_32k_config.json \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--do_train=True \
--train_batch_size=256 \
--num_train_steps=1000000 \
--learning_rate=1e-4 \
--save_checkpoints_steps=100000 \
--keep_checkpoint_max=10 \
--use_tpu=True \
--tpu_name=<tpu name> \
--num_tpu_cores=8

# For mecab-ipadic-char-4k BERT-base models
$ python3 run_pretraining.py \
--input_file="/path/to/pretraining-data.tf_record.*" \
--output_dir="/path/to/output_dir" \
--bert_config_file=bert_base_4k_config.json \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--do_train=True \
--train_batch_size=256 \
--num_train_steps=1000000 \
--learning_rate=1e-4 \
--save_checkpoints_steps=100000 \
--keep_checkpoint_max=10 \
--use_tpu=True \
--tpu_name=<tpu name> \
--num_tpu_cores=8
```

## Related Work

- Original BERT model by Google Research Team
    - https://github.com/google-research/bert
    - https://github.com/tensorflow/models/tree/master/official/nlp/bert (for TensorFlow 2.0)
- Juman-tokenized Japanese BERT model
    - Author: Kurohashi-Kawahara Laboratory, Kyoto University
    - http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT日本語Pretrainedモデル
- Sentencepiece Japanese BERT model
    - Author: Yohei Kikuta
    - https://github.com/yoheikikuta/bert-japanese
- Sentencepiece Japanese BERT model, trained on SNS corpus
    - Author: Hottolink, Inc.
    - https://github.com/hottolink/hottoSNS-bert

## Acknowledgments

For training models, we used Cloud TPUs provided by [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc/) program.
