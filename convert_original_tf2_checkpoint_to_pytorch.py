# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2023 Masatoshi Suzuki (@singletongue)
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

    logger.info("Converting TensorFlow checkpoint from %s", tf_checkpoint_path)

    for full_name, shape in tf.train.list_variables(tf_checkpoint_path):
        pointer = model
        trace = []

        if len(shape) == 0:
            logger.info("Skipping non-tensor variable: %s", full_name)
            continue

        if "optimizer/" in full_name:
            logger.info("Skipping optimizer weights: %s", full_name)
            continue

        split_name = full_name.split("/")
        name = split_name.pop(0)
        if name == "encoder":
            pointer = getattr(pointer, "bert")
            trace.append("bert")

            name = split_name.pop(0)
            if name.startswith("layer_with_weights"):
                layer_num = int(name.split("-")[-1])
                # if layer_num == 0:
                    # word embedding (not saved with tensorflow-models 2.10.0)
                    # trace.extend(["embeddings", "word_embeddings"])
                    # pointer = getattr(pointer, "embeddings")
                    # pointer = getattr(pointer, "word_embeddings")
                if layer_num == 1:
                    # position_embedding
                    trace.extend(["embeddings", "position_embeddings"])
                    pointer = getattr(pointer, "embeddings")
                    pointer = getattr(pointer, "position_embeddings")
                elif layer_num == 2:
                    # type_embeddings
                    trace.extend(["embeddings", "token_type_embeddings"])
                    pointer = getattr(pointer, "embeddings")
                    pointer = getattr(pointer, "token_type_embeddings")
                elif layer_num == 3:
                    # embeddings/layer_norm
                    trace.extend(["embeddings", "LayerNorm"])
                    pointer = getattr(pointer, "embeddings")
                    pointer = getattr(pointer, "LayerNorm")
                elif layer_num >= 4 and layer_num < config.num_hidden_layers + 4:
                    # transformer/layer_x
                    trace.extend(["encoder", "layer", str(layer_num - 4)])
                    pointer = getattr(pointer, "encoder")
                    pointer = getattr(pointer, "layer")
                    pointer = pointer[layer_num - 4]
                elif layer_num == config.num_hidden_layers + 4:
                    # pooler_transform (not trained with tensorflow-models 2.10.0)
                    continue
                else:
                    logger.warning("Skipping unknown weight name: %s", full_name)
                    continue
        elif name == "masked_lm":
            trace.extend(["cls", "predictions"])
            pointer = getattr(pointer, "cls")
            pointer = getattr(pointer, "predictions")

            name = split_name.pop(0)
            if name == "dense":
                trace.extend(["transform", "dense"])
                pointer = getattr(pointer, "transform")
                pointer = getattr(pointer, "dense")
            elif name == "embedding_table":
                trace.extend(["decoder", "weight"])
                pointer = getattr(pointer, "decoder")
                pointer = getattr(pointer, "weight")
            elif name == "layer_norm":
                trace.extend(["transform", "LayerNorm"])
                pointer = getattr(pointer, "transform")
                pointer = getattr(pointer, "LayerNorm")
            elif name == "output_bias.Sbias":
                trace.extend(["bias"])
                pointer = getattr(pointer, "bias")
            else:
                logger.warning("Skipping unknown weight name: %s", full_name)
                continue
        elif name == "model":
            names = split_name[:3]
            split_name = split_name[3:]
            if names == ["classification_heads", "0", "out_proj"]:
                trace.extend(["cls", "seq_relationship"])
                pointer = getattr(pointer, "cls")
                pointer = getattr(pointer, "seq_relationship")
            else:
                logger.warning("Skipping unknown weight name: %s", full_name)
                continue
        elif name == "next_sentence..pooler_dense":
            trace.extend(["bert", "pooler", "dense"])
            pointer = getattr(pointer, "bert")
            pointer = getattr(pointer, "pooler")
            pointer = getattr(pointer, "dense")
        else:
            logger.warning("Skipping unknown weight name: %s", full_name)
            continue

        # iterate over the rest depths
        for name in split_name:
            if name == "_attention_layer":
                # self-attention layer
                trace.append("attention")
                pointer = getattr(pointer, "attention")
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
                trace.extend(["self", "key"])
                pointer = getattr(pointer, "self")
                pointer = getattr(pointer, "key")
            elif name == "_query_dense":
                # attention query
                trace.extend(["self", "query"])
                pointer = getattr(pointer, "self")
                pointer = getattr(pointer, "query")
            elif name == "_value_dense":
                # attention value
                trace.extend(["self", "value"])
                pointer = getattr(pointer, "self")
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
            elif name == "embeddings":
                # embeddins weights
                trace.append("weight")
                pointer = getattr(pointer, "weight")
            elif name == ".ATTRIBUTES":
                # full variable name ends with .ATTRIBUTES/VARIABLE_VALUE
                break
            else:
                logger.warning("Skipping unknown weight name: %s", full_name)

        logger.info("Loading TF weight %s with shape %s", full_name, shape)

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

        logger.info("Successfully set variable %s to PyTorch layer %s", full_name, trace)

        if full_name == "masked_lm/embedding_table/.ATTRIBUTES/VARIABLE_VALUE":
            word_embeddings_pointer = model.bert.embeddings.word_embeddings.weight
            word_embeddings_trace = "bert.embeddings.word_embeddings.weight"

            if word_embeddings_pointer.shape == array.shape:
                word_embeddings_pointer.data = torch.from_numpy(array)
            else:
                raise ValueError(
                    f"Shape mismatch in layer {full_name}: "
                    f"Model expects shape {word_embeddings_pointer.shape} but layer contains shape: {array.shape}"
                )

            logger.info("Successfully set variable %s to PyTorch layer %s", full_name, word_embeddings_trace)

    return model


def convert_tf2_checkpoint_to_pytorch(tf_checkpoint_path, config_path, pytorch_dump_path):
    # Initialize PyTorch model
    logger.info("Loading model based on config from %s...", config_path)
    config = BertConfig.from_json_file(config_path)
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    logger.info("Loading weights from checkpoint %s...", tf_checkpoint_path)
    load_tf2_weights_in_bert(model, tf_checkpoint_path, config)

    # Save pytorch-model
    logger.info("Saving PyTorch model to %s...", pytorch_dump_path)
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
