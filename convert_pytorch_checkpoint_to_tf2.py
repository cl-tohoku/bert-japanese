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
""" Convert pytorch checkpoints to TensorFlow """

import argparse
import logging

import numpy as np
import torch
from transformers import (
    AlbertConfig,
    AlbertForPreTraining,
    BertConfig,
    BertForPreTraining,
    TFAlbertForPreTraining,
    TFBertForPreTraining,
    load_pytorch_checkpoint_in_tf2_model,
)


logging.basicConfig(level=logging.INFO)

MODEL_CLASSES = {
    "bert": (
        BertConfig,
        TFBertForPreTraining,
        BertForPreTraining,
    ),
    "albert": (
        AlbertConfig,
        TFAlbertForPreTraining,
        AlbertForPreTraining,
    ),
}

def convert_pt_checkpoint_to_tf(
    model_type,
    pytorch_checkpoint_path,
    config_file,
    tf_dump_path,
    compare_with_pt_model=False,
):
    if model_type not in MODEL_CLASSES:
        raise ValueError("Unrecognized model type, should be one of {}.".format(list(MODEL_CLASSES.keys())))

    config_class, model_class, pt_model_class = MODEL_CLASSES[model_type]

    # Initialise TF model
    config = config_class.from_json_file(config_file)
    config.output_hidden_states = True
    config.output_attentions = True
    print("Building TensorFlow model from configuration: {}".format(str(config)))
    tf_model = model_class(config)

    # Load PyTorch checkpoint in tf2 model:
    tf_model = load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path)

    if compare_with_pt_model:
        tfo = tf_model(tf_model.dummy_inputs, training=False)  # build the network

        state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
        pt_model = pt_model_class.from_pretrained(pretrained_model_name_or_path=None,
                                                  config=config,
                                                  state_dict=state_dict)

        with torch.no_grad():
            pto = pt_model(**pt_model.dummy_inputs)

        np_pt = pto[0].numpy()
        np_tf = tfo[0].numpy()
        diff = np.amax(np.abs(np_pt - np_tf))
        print("Max absolute difference between models outputs {}".format(diff))
        assert diff <= 2e-2, "Error, model absolute difference is >2e-2: {}".format(diff)

    # Save pytorch-model
    print("Save TensorFlow model to {}".format(tf_dump_path))
    tf_model.save_weights(tf_dump_path, save_format="h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_dump_path",
        type=str,
        required=True,
        help="Path to the output Tensorflow dump file."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type selected in the list of {}. If not given, will download and convert all the models from AWS.".format(
            list(MODEL_CLASSES.keys())
        ),
    )
    parser.add_argument(
        "--pytorch_checkpoint_path",
        type=str,
        required=True,
        help="Path to the PyTorch checkpoint path or shortcut name to download from AWS."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model."
    )
    parser.add_argument(
        "--compare_with_pt_model",
        action="store_true",
        help="Compare Tensorflow and PyTorch model predictions."
    )
    args = parser.parse_args()

    convert_pt_checkpoint_to_tf(args.model_type.lower(),
                                args.pytorch_checkpoint_path,
                                args.config_file,
                                args.tf_dump_path,
                                compare_with_pt_model=args.compare_with_pt_model)
