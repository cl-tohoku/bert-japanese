# Pretrained Japanese BERT models

This is a repository of pretrained Japanese BERT models.
The models are available in [Transformers](https://github.com/huggingface/transformers) by Hugging Face.

- Model hub: https://huggingface.co/cl-tohoku

For information on the previous versions of our pretrained models, see the [v1.0](https://github.com/cl-tohoku/bert-japanese/tree/v1.0) tag of this repository.

## Model Architecture

The architecture of our models are the same as the original BERT models proposed by Google.
- **BERT-base** models consist of 12 layers, 768 dimensions of hidden states, and 12 attention heads.
- **BERT-large** models consist of 24 layers, 1024 dimensions of hidden states, and 16 attention heads.

## Training Data

The models are trained on the Japanese version of Wikipedia.
The training corpus is generated from the Wikipedia Cirrussearch dump file as of August 31, 2020.

The generated corpus files are 4.0GB in total, consisting of approximately 30M sentences.
We used the [MeCab](https://taku910.github.io/mecab/) morphological parser with [mecab-ipadic-NEologd](https://github.com/neologd/mecab-ipadic-neologd) dictionary to split texts into sentences.

```sh
$ WORK_DIR="$HOME/work/bert-japanese"

$ python make_corpus_wiki.py \
--input_file jawiki-20200831-cirrussearch-content.json.gz \
--output_file $WORK_DIR/corpus/jawiki-20200831/corpus.txt \
--min_text_length 10 \
--max_text_length 200 \
--mecab_option "-r $HOME/local/etc/mecabrc -d $HOME/local/lib/mecab/dic/mecab-ipadic-neologd-v0.0.7"

# Split corpus files for parallel preprocessing of the files
$ python merge_split_corpora.py \
--input_files $WORK_DIR/corpus/jawiki-20200831/corpus.txt \
--output_dir $WORK_DIR/corpus/jawiki-20200831 \
--num_files 8

# Sample some lines for training tokenizers
$ cat $WORK_DIR/corpus/jawiki-20200831/corpus.txt|grep -v '^$'|shuf|head -n 1000000 \
> $WORK_DIR/corpus/jawiki-20200831/corpus_sampled.txt
```

## Tokenization

For each of BERT-base and BERT-large, we provide two models with different tokenization methods.

- For **`wordpiece`** models, the texts are first tokenized by MeCab with the Unidic 2.1.2 dictionary and then split into subwords by the WordPiece algorithm.
  The vocabulary size is 32768.
- For **`character`** models, the texts are first tokenized by MeCab with the Unidic 2.1.2 dictionary and then split into characters.
  The vocabulary size is 6144.

We used [`fugashi`](https://github.com/polm/fugashi) and [`unidic-lite`](https://github.com/polm/unidic-lite) packages for the tokenization.

```sh
$ WORK_DIR="$HOME/work/bert-japanese"

# WordPiece (unidic_lite)
$ TOKENIZERS_PARALLELISM=false python train_tokenizer.py \
--input_files $WORK_DIR/corpus/jawiki-20200831/corpus_sampled.txt \
--output_dir $WORK_DIR/tokenizers/jawiki-20200831/wordpiece_unidic_lite \
--tokenizer_type wordpiece \
--mecab_dic_type unidic_lite \
--vocab_size 32768 \
--limit_alphabet 6129 \
--num_unused_tokens 10

# Character
$ head -n 6144 $WORK_DIR/tokenizers/jawiki-20200831/wordpiece_unidic_lite/vocab.txt \
> $WORK_DIR/tokenizers/jawiki-20200831/character/vocab.txt
```

## Training

The models are trained with the same configuration as the original BERT; 512 tokens per instance, 256 instances per batch, and 1M training steps.
For training of the MLM (masked language modeling) objective, we introduced **whole word masking** in which all of the subword tokens corresponding to a single word (tokenized by MeCab) are masked at once.

For training of each model, we used a v3-8 instance of Cloud TPUs provided by [TensorFlow Research Cloud program](https://www.tensorflow.org/tfrc/).
The training took about 5 days and 14 days for BERT-base and BERT-large models, respectively.

### Creation of the pretraining data

```sh
$ WORK_DIR="$HOME/work/bert-japanese"

# WordPiece (unidic_lite)
$ mkdir -p $WORK_DIR/bert/jawiki-20200831/wordpiece_unidic_lite/pretraining_data
# It takes 3h and 420GB RAM, producing 43M instances
$ seq -f %02g 1 8|xargs -L 1 -I {} -P 8 python create_pretraining_data.py \
--input_file $WORK_DIR/corpus/jawiki-20200831/corpus_{}.txt \
--output_file $WORK_DIR/bert/jawiki-20200831/wordpiece_unidic_lite/pretraining_data/pretraining_data_{}.tfrecord.gz \
--vocab_file $WORK_DIR/tokenizers/jawiki-20200831/wordpiece_unidic_lite/vocab.txt \
--tokenizer_type wordpiece \
--mecab_dic_type unidic_lite \
--do_whole_word_mask \
--gzip_compress \
--max_seq_length 512 \
--max_predictions_per_seq 80 \
--dupe_factor 10

# Character
$ mkdir $WORK_DIR/bert/jawiki-20200831/character/pretraining_data
# It takes 4h10m and 615GB RAM, producing 55M instances
$ seq -f %02g 1 8|xargs -L 1 -I {} -P 8 python create_pretraining_data.py \
--input_file $WORK_DIR/corpus/jawiki-20200831/corpus_{}.txt \
--output_file $WORK_DIR/bert/jawiki-20200831/character/pretraining_data/pretraining_data_{}.tfrecord.gz \
--vocab_file $WORK_DIR/tokenizers/jawiki-20200831/character/vocab.txt \
--tokenizer_type character \
--mecab_dic_type unidic_lite \
--do_whole_word_mask \
--gzip_compress \
--max_seq_length 512 \
--max_predictions_per_seq 80 \
--dupe_factor 10
```

### Training of the models

**Note:** all the necessary files need to be stored in a Google Cloud Storage (GCS) bucket.

```sh
# BERT-base, WordPiece (unidic_lite)
$ ctpu up -name tpu01 -tpu-size v3-8 -tf-version 2.3
$ cd /usr/share/models
$ sudo pip3 install -r official/requirements.txt
$ tmux
$ export PYTHONPATH="$PYTHONPATH:/usr/share/tpu/models"
$ WORK_DIR="gs://<your GCS bucket name>/bert-japanese"
$ python3 official/nlp/bert/run_pretraining.py \
--input_files="$WORK_DIR/bert/jawiki-20200831/wordpiece_unidic_lite/pretraining_data/pretraining_data_*.tfrecord" \
--model_dir="$WORK_DIR/bert/jawiki-20200831/wordpiece_unidic_lite/bert-base" \
--bert_config_file="$WORK_DIR/bert/jawiki-20200831/wordpiece_unidic_lite/bert-base/config.json" \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--train_batch_size=256 \
--learning_rate=1e-4 \
--num_train_epochs=100 \
--num_steps_per_epoch=10000 \
--optimizer_type=adamw \
--warmup_steps=10000 \
--distribution_strategy=tpu \
--tpu=tpu01

# BERT-base, Character
$ ctpu up -name tpu02 -tpu-size v3-8 -tf-version 2.3
$ cd /usr/share/models
$ sudo pip3 install -r official/requirements.txt
$ tmux
$ export PYTHONPATH="$PYTHONPATH:/usr/share/tpu/models"
$ WORK_DIR="gs://<your GCS bucket name>/bert-japanese"
$ python3 official/nlp/bert/run_pretraining.py \
--input_files="$WORK_DIR/bert/jawiki-20200831/character/pretraining_data/pretraining_data_*.tfrecord" \
--model_dir="$WORK_DIR/bert/jawiki-20200831/character/bert-base" \
--bert_config_file="$WORK_DIR/bert/jawiki-20200831/character/bert-base/config.json" \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--train_batch_size=256 \
--learning_rate=1e-4 \
--num_train_epochs=100 \
--num_steps_per_epoch=10000 \
--optimizer_type=adamw \
--warmup_steps=10000 \
--distribution_strategy=tpu \
--tpu=tpu02

# BERT-large, WordPiece (unidic_lite)
$ ctpu up -name tpu03 -tpu-size v3-8 -tf-version 2.3
$ cd /usr/share/models
$ sudo pip3 install -r official/requirements.txt
$ tmux
$ export PYTHONPATH="$PYTHONPATH:/usr/share/tpu/models"
$ WORK_DIR="gs://<your GCS bucket name>/bert-japanese"
$ python3 official/nlp/bert/run_pretraining.py \
--input_files="$WORK_DIR/bert/jawiki-20200831/wordpiece_unidic_lite/pretraining_data/pretraining_data_*.tfrecord" \
--model_dir="$WORK_DIR/bert/jawiki-20200831/wordpiece_unidic_lite/bert-large" \
--bert_config_file="$WORK_DIR/bert/jawiki-20200831/wordpiece_unidic_lite/bert-large/config.json" \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--train_batch_size=256 \
--learning_rate=5e-5 \
--num_train_epochs=100 \
--num_steps_per_epoch=10000 \
--optimizer_type=adamw \
--warmup_steps=10000 \
--distribution_strategy=tpu \
--tpu=tpu03

# BERT-large, Character
$ ctpu up -name tpu04 -tpu-size v3-8 -tf-version 2.3
$ cd /usr/share/models
$ sudo pip3 install -r official/requirements.txt
$ tmux
$ export PYTHONPATH="$PYTHONPATH:/usr/share/tpu/models"
$ WORK_DIR="gs://<your GCS bucket name>/bert-japanese"
$ python3 official/nlp/bert/run_pretraining.py \
--input_files="$WORK_DIR/bert/jawiki-20200831/character/pretraining_data/pretraining_data_*.tfrecord" \
--model_dir="$WORK_DIR/bert/jawiki-20200831/character/bert-large" \
--bert_config_file="$WORK_DIR/bert/jawiki-20200831/character/bert-large/config.json" \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--train_batch_size=256 \
--learning_rate=5e-5 \
--num_train_epochs=100 \
--num_steps_per_epoch=10000 \
--optimizer_type=adamw \
--warmup_steps=10000 \
--distribution_strategy=tpu \
--tpu=tpu04
```

## Licenses

The pretrained models are distributed under the terms of the [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

The codes in this repository are distributed under the Apache License 2.0.

## Related Work

- Original BERT model by Google Research Team
    - https://github.com/google-research/bert
    - https://github.com/tensorflow/models/tree/master/official/nlp/bert (for TensorFlow 2.0)
- Juman-tokenized Japanese BERT model
    - Author: Kurohashi-Kawahara Laboratory, Kyoto University
    - http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT日本語Pretrainedモデル
- MeCab-Jumandic-tokenized Japanese BERT model (trained with a large mini-batch size)
    - Author: National Institute of Information and Communications Technology (NICT)
    - https://alaginrc.nict.go.jp/nict-bert/index.html
- Sentencepiece Japanese BERT model
    - Author: Yohei Kikuta
    - https://github.com/yoheikikuta/bert-japanese
- Sentencepiece Japanese BERT model, trained on SNS corpus
    - Author: Hottolink, Inc.
    - https://github.com/hottolink/hottoSNS-bert

## Acknowledgments

The models are trained with Cloud TPUs provided by [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc/) program.
