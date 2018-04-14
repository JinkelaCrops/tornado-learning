# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

r"""Decode from trained T2T models.

This binary performs inference using the Estimator API.

Example usage to decode from dataset:

  t2t-decoder \
      --data_dir ~/data \
      --problems=algorithmic_identity_binary40 \
      --model=transformer
      --hparams_set=transformer_base

Set FLAGS.decode_interactive or FLAGS.decode_from_file for alternative decode
sources.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config

import os
from app.loggerInst import log
import sys

sys.path.extend(["/home/tmxmall/PycharmProjects/tensor2tensor"])
try:
    file_name = __file__.split("/")[-1]
except:
    file_name = "my_t2t_decoder_active"

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_DEVICE

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils.decoding import log_decode_results

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# Additional flags in bin/t2t_trainer.py and utils/flags.py
flags.DEFINE_string("decode_from_file", None,
                    "Path to the source file for decoding")
flags.DEFINE_string("decode_to_file", None,
                    "Path to the decoded (output) file")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")


def create_hparams():
    return trainer_lib.create_hparams(
        FLAGS.hparams_set,
        FLAGS.hparams,
        data_dir=os.path.expanduser(FLAGS.data_dir),
        problem_name=FLAGS.problems)


def create_decode_hparams():
    decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
    decode_hp.add_hparam("shards", FLAGS.decode_shards)
    decode_hp.add_hparam("shard_id", FLAGS.worker_id)
    return decode_hp


def decode(estimator, hparams, decode_hp):
    if FLAGS.decode_interactive:
        decoding.decode_interactively(estimator, hparams, decode_hp)
    elif FLAGS.decode_from_file:
        decoding.decode_from_file(estimator, FLAGS.decode_from_file, hparams,
                                  decode_hp, FLAGS.decode_to_file)
    else:
        decoding.decode_from_dataset(
            estimator,
            FLAGS.problems.split("-"),
            hparams,
            decode_hp,
            decode_to_file=FLAGS.decode_to_file,
            dataset_split="test" if FLAGS.eval_use_test_set else None)


FLAGS.data_dir = config.VOCAB_DIR
FLAGS.problems = config.PROBLEM_NAME
FLAGS.model = config.MODEL_NAME
FLAGS.hparams_set = config.HPARAMS_SET
FLAGS.output_dir = config.MODEL_DIR
FLAGS.decode_hparams = config.DECODE_HPARAMS

tf.logging.set_verbosity(tf.logging.ERROR)
FLAGS.use_tpu = False  # decoding not supported on TPU


class SessFieldPredict(object):

    def __init__(self, batch_size):
        self.hparams = create_hparams()
        self.encoders = registry.problem(FLAGS.problems).feature_encoders(FLAGS.data_dir)
        self.ckpt = tf.train.get_checkpoint_state(FLAGS.output_dir).model_checkpoint_path

        self.inputs_ph = tf.placeholder(shape=(batch_size, None), dtype=tf.int32)  # Just length dimension.
        self.batch_inputs = tf.reshape(self.inputs_ph, [batch_size, -1, 1, 1])  # Make it 4D.
        self.features = {"inputs": self.batch_inputs}
        # Prepare the model and the graph when model runs on features.
        log.info(f"[{file_name}] SessFieldPredict: register T2TModel")
        self.model = registry.model(FLAGS.model)(self.hparams, tf.estimator.ModeKeys.PREDICT)
        self.model_spec = self.model.estimator_spec_predict(self.features)
        self.prediction = self.model_spec.predictions

        self.inputs_vocab = self.hparams.problems[0].vocabulary["inputs"]
        self.targets_vocab = self.hparams.problems[0].vocabulary["targets"]
        self.problem_name = FLAGS.problems

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.GPU_MEM_FRAC)
        self.sess_config = tf.ConfigProto(gpu_options=gpu_options)

        self.batch_size = batch_size
        log.info(f"[{file_name}] SessFieldPredict: registered")



def main(_):
    FLAGS.decode_interactive = True
    hp = create_hparams()
    decode_hp = create_decode_hparams()

    estimator = trainer_lib.create_estimator(
        FLAGS.model,
        hp,
        t2t_trainer.create_run_config(hp),
        decode_hparams=decode_hp,
        use_tpu=False)

    decode(estimator, hp, decode_hp)


if __name__ == "__main__":
    tf.app.run()
