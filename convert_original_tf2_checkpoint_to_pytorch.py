# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""
This script can be used to convert a head-less TF2.x Bert model to PyTorch,
as published on the official GitHub: https://github.com/tensorflow/models/tree/master/official/nlp/bert

TF2.x uses different variable names from the original BERT (TF 1.4) implementation.
The script re-maps the TF2.x Bert weight names to the original names, so the model can be imported with Huggingface/transformer.

You may adapt this script to include classification/MLM/NSP/etc. heads.
"""
import argparse
import logging
import os
import re

import tensorflow as tf
import torch

from transformers import BertConfig, BertForPreTraining


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tf2_weights_in_bert(model, tf_checkpoint_path, config):
    tf_checkpoint_path = os.path.abspath(tf_checkpoint_path)

    logger.info("Converting TensorFlow checkpoint from {}".format(tf_checkpoint_path))

    for full_name, shape in tf.train.list_variables(tf_checkpoint_path):
        pointer = model
        trace = []

        if "optimizer/" in full_name:
            logger.info("Skipping optimizer weights: {}".format(full_name))
            continue

        split_name = full_name.split("/")
        name = split_name.pop(0)
        if name != "model":
            logger.info("Skipping: {}".format(full_name))
            continue

        name = split_name.pop(0)
        if name != "layer_with_weights-0":
            logger.warning(f"Skipping unknown weight name: {full_name}")
            continue

        logger.info("Loading TF weight {} with shape {}".format(full_name, shape))

        name = split_name.pop(0)
        if name == "layer_with_weights-0":
            # transformer encoder

            pointer = getattr(pointer, "bert")
            trace.append("bert")

            name = split_name.pop(0)
            if name.startswith("layer_with_weights"):
                layer_num = int(name.split("-")[-1])
                if layer_num <= 2:
                    # embedding layers
                    if layer_num == 0:
                        trace.extend(["embeddings", "word_embeddings"])
                        pointer = getattr(pointer, "embeddings")
                        pointer = getattr(pointer, "word_embeddings")
                    elif layer_num == 1:
                        trace.extend(["embeddings", "position_embeddings"])
                        pointer = getattr(pointer, "embeddings")
                        pointer = getattr(pointer, "position_embeddings")
                    else:
                        trace.extend(["embeddings", "token_type_embeddings"])
                        pointer = getattr(pointer, "embeddings")
                        pointer = getattr(pointer, "token_type_embeddings")
                    trace.append("weight")
                    pointer = getattr(pointer, "weight")
                elif layer_num == 3:
                    # embedding LayerNorm
                    trace.extend(["embeddings", "LayerNorm"])
                    pointer = getattr(pointer, "embeddings")
                    pointer = getattr(pointer, "LayerNorm")
                elif layer_num >= 4 and layer_num < config.num_hidden_layers + 4:
                    # encoder layers
                    trace.extend(["encoder", "layer", str(layer_num - 4)])
                    pointer = getattr(pointer, "encoder")
                    pointer = getattr(pointer, "layer")
                    pointer = pointer[layer_num - 4]
                elif layer_num == config.num_hidden_layers + 4:
                    # pooler layer
                    trace.extend(["pooler", "dense"])
                    pointer = getattr(pointer, "pooler")
                    pointer = getattr(pointer, "dense")
        elif name == "layer_with_weights-1":
            # next sentence prediction

            trace.extend(["cls", "seq_relationship"])
            pointer = getattr(pointer, "cls")
            pointer = getattr(pointer, "seq_relationship")
        elif name == "layer_with_weights-2":
            # masked lm

            trace.extend(["cls", "predictions"])
            pointer = getattr(pointer, "cls")
            pointer = getattr(pointer, "predictions")

            name = split_name.pop(0)
            if name == "dense":
                trace.extend(["transform", "dense"])
                pointer = getattr(pointer, "transform")
                pointer = getattr(pointer, "dense")
            elif name == "layer_norm":
                trace.extend(["transform", "LayerNorm"])
                pointer = getattr(pointer, "transform")
                pointer = getattr(pointer, "LayerNorm")
            elif name == "embedding_table":
                trace.extend(["decoder", "weight"])
                pointer = getattr(pointer, "decoder")
                pointer = getattr(pointer, "weight")
            elif name == "output_bias.Sbias":
                trace.extend(["bias"])
                pointer = getattr(pointer, "bias")
        else:
            logger.warning(f"Skipping unknown weight name: {full_name}")
            continue

        # iterate over the rest depths
        for name in split_name:
            if name == "_attention_layer":
                # self-attention layer
                trace.extend(["attention", "self"])
                pointer = getattr(pointer, "attention")
                pointer = getattr(pointer, "self")
            elif name == "_attention_layer_norm":
                # output attention norm
                trace.extend(["attention", "output", "LayerNorm"])
                pointer = getattr(pointer, "attention")
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "LayerNorm")
            elif name == "_attention_output_dense":
                # output attention dense
                trace.extend(["attention", "output", "dense"])
                pointer = getattr(pointer, "attention")
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "dense")
            elif name == "_intermediate_dense":
                # attention intermediate dense
                trace.extend(["intermediate", "dense"])
                pointer = getattr(pointer, "intermediate")
                pointer = getattr(pointer, "dense")
            elif name == "_output_dense":
                # output dense
                trace.extend(["output", "dense"])
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "dense")
            elif name == "_output_layer_norm":
                # output dense
                trace.extend(["output", "LayerNorm"])
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "LayerNorm")
            elif name == "_key_dense":
                # attention key
                trace.append("key")
                pointer = getattr(pointer, "key")
            elif name == "_query_dense":
                # attention query
                trace.append("query")
                pointer = getattr(pointer, "query")
            elif name == "_value_dense":
                # attention value
                trace.append("value")
                pointer = getattr(pointer, "value")
            elif name == "dense":
                # attention value
                trace.append("dense")
                pointer = getattr(pointer, "dense")
            elif name in ["bias", "beta"]:
                # norm biases
                trace.append("bias")
                pointer = getattr(pointer, "bias")
            elif name in ["kernel", "gamma"]:
                # norm weights
                trace.append("weight")
                pointer = getattr(pointer, "weight")
            elif name == ".ATTRIBUTES":
                # full variable name ends with .ATTRIBUTES/VARIABLE_VALUE
                break
            else:
                logger.warning(f"Skipping unknown weight name: {full_name}")

        array = tf.train.load_variable(tf_checkpoint_path, full_name)

        # for certain layers reshape is necessary
        trace = ".".join(trace)
        if re.match(r"(\S+)\.attention\.self\.(key|value|query)\.(bias|weight)", trace) or \
           re.match(r"(\S+)\.attention\.output\.dense\.weight", trace):
            array = array.reshape(pointer.data.shape)
        if "kernel" in full_name:
            array = array.transpose()

        if pointer.shape == array.shape:
            pointer.data = torch.from_numpy(array)
        else:
            raise ValueError(
                f"Shape mismatch in layer {full_name}: "
                f"Model expects shape {pointer.shape} but layer contains shape: {array.shape}"
            )

        logger.info(f"Successfully set variable {full_name} to PyTorch layer {trace}")

    return model


def convert_tf2_checkpoint_to_pytorch(tf_checkpoint_path, config_path, pytorch_dump_path):
    # Initialize PyTorch model
    logger.info(f"Loading model based on config from {config_path}...")
    config = BertConfig.from_json_file(config_path)
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    logger.info(f"Loading weights from checkpoint {tf_checkpoint_path}...")
    load_tf2_weights_in_bert(model, tf_checkpoint_path, config)

    # Save pytorch-model
    logger.info(f"Saving PyTorch model to {pytorch_dump_path}...")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow 2.x checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model (must include filename).",
    )
    args = parser.parse_args()
    convert_tf2_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)
