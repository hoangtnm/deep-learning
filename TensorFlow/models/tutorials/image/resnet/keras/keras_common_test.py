"""Tests for the keras_common module."""
from mock import Mock
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

import keras_common

tf.logging.set_verbosity(tf.logging.ERROR)

class KerasCommonTests(tf.test.TestCase):
    """Tests for keras_common."""

    @classmethod
    def setUpClass(cls):  # pylint: disable=invalid-name
        super(KerasCommonTests, cls).setUpClass()

    def test_build_stats(self):

        history = self._build_history(1.145, cat_accuracy=.99988)
        eval_output = self._build_eval_output(.56432111, 5.990)
        stats = keras_common.build_stats(history, eval_output)

        self.assertEqual(1.145, stats['loss'])
        self.assertEqual(.99988, stats['training_accuracy_top_1'])

        self.assertEqual(.56432111, stats['accuracy_top_1'])
        self.assertEqual(5.990, stats['eval_loss'])

    def test_build_stats_sparse(self):

        history = self._build_history(1.145, cat_accuracy_sparse=.99988)
        eval_output = self._build_eval_output(.928, 1.9844)
        stats = keras_common.build_stats(history, eval_output)

        self.assertEqual(1.145, stats['loss'])
        self.assertEqual(.99988, stats['training_accuracy_top_1'])

        self.assertEqual(.928, stats['accuracy_top_1'])
        self.assertEqual(1.9844, stats['eval_loss'])

    def _build_history(self, loss, cat_accuracy=None,
                       cat_accuracy_sparse=None):
        history_p = Mock()
        history = {}
        history_p.history = history
        history['loss'] = [np.float64(loss)]
        if cat_accuracy:
            history['categorical_accuracy'] = [np.float64(cat_accuracy)]
        if cat_accuracy_sparse:
            history['sparse_categorical_accuracy'] = [np.float64(cat_accuracy_sparse)]

        return history_p

    def _build_eval_output(self, top_1, eval_loss):
        eval_output = [np.float64(eval_loss), np.float64(top_1)]
        return eval_output
