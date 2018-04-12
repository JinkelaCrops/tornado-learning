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

import os

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder

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


def score_file(filename):
    """Score each line in a file and return the scores."""
    # Prepare model.
    hparams = create_hparams()
    encoders = registry.problem(FLAGS.problems).feature_encoders(FLAGS.data_dir)
    has_inputs = "inputs" in encoders

    # Prepare features for feeding into the model.
    if has_inputs:
        inputs_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
        batch_inputs = tf.reshape(inputs_ph, [1, -1, 1, 1])  # Make it 4D.
    features = {"inputs": batch_inputs}
    # Prepare the model and the graph when model runs on features.
    model = registry.model(FLAGS.model)(hparams, tf.estimator.ModeKeys.PREDICT)
    model_spec = model.estimator_spec_predict(features)
    prediction = model_spec.predictions

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Load weights from checkpoint.
        ckpts = tf.train.get_checkpoint_state(FLAGS.output_dir)
        ckpt = ckpts.model_checkpoint_path
        saver.restore(sess, ckpt)
        # Run on each line.
        results = []
        for line in open(filename):
            # tab_split = line.split("\t")
            # if len(tab_split) > 2:
            #     raise ValueError("Each line must have at most one tab separator.")
            # if len(tab_split) == 1:
            #     targets = tab_split[0].strip()
            # else:
            #     targets = tab_split[1].strip()
            #     inputs = tab_split[0].strip()
            inputs = "我 爱 北京 天安门"

            # # Run encoders and append EOS symbol.
            # targets_numpy = encoders["targets"].encode(
            #     targets) + [text_encoder.EOS_ID]
            if has_inputs:
                inputs_numpy = encoders["inputs"].encode(inputs) + [text_encoder.EOS_ID]
            feed = {inputs_ph: inputs_numpy}
            # # Prepare the feed.
            # feed = {
            #     inputs_ph: inputs_numpy,
            #     targets_ph: targets_numpy
            # } if has_inputs else {targets_ph: targets_numpy}
            # Get the score.
            np_loss = sess.run(prediction, feed)
            results.append(np_loss)
    return results


def main(_):
    HOMEPATH = "/media/yanpan/7D4CF1590195F939/Projects/t2t_med"

    FLAGS.data_dir = f"{HOMEPATH}/t2t_data/new_medicine"
    FLAGS.problems = "translate_zhen_new_med_simple"
    FLAGS.model = "transformer"
    FLAGS.hparams_set = "transformer_base_single_gpu_batch_size_4096"
    FLAGS.output_dir = f"{HOMEPATH}/t2t_train/new_medicine/{FLAGS.problems}/{FLAGS.model}-{FLAGS.hparams_set}"
    FLAGS.decode_hparams = "beam_size=4,alpha=0.6"

    tf.logging.set_verbosity(tf.logging.INFO)
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    FLAGS.use_tpu = False  # decoding not supported on TPU

    hp = create_hparams()
    decode_hp = create_decode_hparams()

    estimator = trainer_lib.create_estimator(
        FLAGS.model,
        hp,
        t2t_trainer.create_run_config(hp),
        decode_hparams=decode_hp,
        use_tpu=False)

    # decode(estimator, hp, decode_hp)
    ##############################################################
    from tensor2tensor.utils.decoding import _interactive_input_tensor_to_features_dict
    from tensor2tensor.utils.decoding import _interactive_input_fn
    from tensor2tensor.utils.decoding import make_input_fn_from_generator
    ##############################################################
    hparams = hp

    def input_fn():
        gen_fn = make_input_fn_from_generator(_interactive_input_fn(hparams))
        example = gen_fn()
        example = _interactive_input_tensor_to_features_dict(example, hparams)
        return example

    result_iter = estimator.predict(input_fn)
    for result in result_iter:
        problem_idx = result["problem_choice"]
        is_image = False  # TODO(lukaszkaiser): find out from problem id / class.
        targets_vocab = hparams.problems[problem_idx].vocabulary["targets"]

        if decode_hp.return_beams:
            beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
            scores = None
            if "scores" in result:
                scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
            for k, beam in enumerate(beams):
                tf.logging.info("BEAM %d:" % k)
                beam_string = targets_vocab.decode(_save_until_eos(beam, is_image))
                if scores is not None:
                    tf.logging.info("\"%s\"\tScore:%f" % (beam_string, scores[k]))
                else:
                    tf.logging.info("\"%s\"" % beam_string)
        else:
            if decode_hp.identity_output:
                tf.logging.info(" ".join(map(str, result["outputs"].flatten())))
            else:
                tf.logging.info(
                    targets_vocab.decode(_save_until_eos(result["outputs"], is_image)))

    ###################################################################################
    from tensor2tensor.utils.decoding import _decode_batch_input_fn
    from tensor2tensor.utils.decoding import make_input_fn_from_generator
    from tensor2tensor.utils.decoding import _decode_input_tensor_to_features_dict
    from tensor2tensor.utils.decoding import log_decode_results
    ###################################################################################
    filename = FLAGS.decode_from_file
    hparams = hp
    decode_to_file = FLAGS.decode_to_file
    """Compute predictions on entries in filename and write them out."""
    if not decode_hp.batch_size:
        decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

    problem_id = decode_hp.problem_idx
    # Inputs vocabulary is set to targets if there are no inputs in the problem,
    # e.g., for language models where the inputs are just a prefix of targets.
    has_input = "inputs" in hparams.problems[problem_id].vocabulary
    inputs_vocab_key = "inputs" if has_input else "targets"
    inputs_vocab = hparams.problems[problem_id].vocabulary[inputs_vocab_key]
    targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
    problem_name = FLAGS.problems.split("-")[problem_id]
    sorted_inputs = ["我的 哇 三 大 文化"]
    num_decode_batches = 1  # (len(sorted_inputs) - 1) // decode_hp.batch_size + 1

    def input_fn():

        input_gen = _decode_batch_input_fn(
            problem_id, num_decode_batches, sorted_inputs, inputs_vocab,
            decode_hp.batch_size, decode_hp.max_input_size)

        gen_fn = make_input_fn_from_generator(input_gen)
        example = gen_fn()
        return _decode_input_tensor_to_features_dict(example, hparams)

    decodes = []
    # result_iter = estimator.predict(input_fn)
    # ------------------------------------------------------------------------
    #########################
    from tensorflow.python.training import saver
    from tensorflow.python.training import training
    from tensorflow.python.framework import random_seed
    from tensorflow.python.estimator import model_fn as model_fn_lib
    import six
    #########################
    hooks = []
    checkpoint_path = None
    # Check that model has been trained.
    if not checkpoint_path:
        checkpoint_path = saver.latest_checkpoint(estimator._model_dir)
    if not checkpoint_path:
        raise ValueError('Could not find trained model in model_dir: {}.'.format(
            estimator._model_dir))

    random_seed.set_random_seed(estimator._config.tf_random_seed)
    # estimator._create_and_assert_global_step(g)
    features, input_hooks = estimator._get_features_from_input_fn(
        input_fn, model_fn_lib.ModeKeys.PREDICT)

    def _get_features_from_input_fn(input_fn, mode):
        """Extracts the `features` from return values of `input_fn`."""
        result = self._call_input_fn(input_fn, mode)
        input_hooks = []
        if isinstance(result, dataset_ops.Dataset):
            iterator = result.make_initializable_iterator()
            input_hooks.append(_DatasetInitializerHook(iterator))
            result = iterator.get_next()
        if isinstance(result, (list, tuple)):
            # Unconditionally drop the label (the second element of result).
            result = result[0]

        if not _has_dataset_or_queue_runner(result):
            logging.warning('Input graph does not use tf.data.Dataset or contain a '
                            'QueueRunner. That means predict yields forever. '
                            'This is probably a mistake.')
        return result, input_hooks

    estimator_spec = estimator._call_model_fn(
        features, None, model_fn_lib.ModeKeys.PREDICT, estimator.config)
    predictions = estimator._extract_keys(estimator_spec.predictions, None)  # predict_keys is None

    mon_sess = training.MonitoredSession(
        session_creator=training.ChiefSessionCreator(
            checkpoint_filename_with_path=checkpoint_path,
            scaffold=estimator_spec.scaffold,
            config=estimator._session_config),
        hooks=input_hooks + hooks)

    preds_evaluated = mon_sess.run(predictions)

    preds = []
    for i in range(estimator._extract_batch_length(preds_evaluated)):
        pred_dict_tmp = {
            key: value[i]
            for key, value in six.iteritems(preds_evaluated)
        }
        preds.append(pred_dict_tmp)

        mon_sess.close()

    # ------------------------------------------------------------------------
    result_iter = preds

    for result in result_iter:
        if decode_hp.return_beams:
            beam_decodes = []
            beam_scores = []
            output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
            scores = None
            if "scores" in result:
                scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
            for k, beam in enumerate(output_beams):
                tf.logging.info("BEAM %d:" % k)
                score = scores and scores[k]
                decoded_outputs, _ = log_decode_results(result["inputs"], beam,
                                                        problem_name, None,
                                                        inputs_vocab, targets_vocab)
                beam_decodes.append(decoded_outputs)
                if decode_hp.write_beam_scores:
                    beam_scores.append(score)
            if decode_hp.write_beam_scores:
                decodes.append("\t".join(
                    ["\t".join([d, "%.2f" % s]) for d, s
                     in zip(beam_decodes, beam_scores)]))
            else:
                decodes.append("\t".join(beam_decodes))
        else:
            decoded_outputs, _ = log_decode_results(result["inputs"],
                                                    result["outputs"], problem_name,
                                                    None, inputs_vocab, targets_vocab)
            decodes.append(decoded_outputs)
    print(decodes)


if __name__ == "__main__":
    tf.app.run()
