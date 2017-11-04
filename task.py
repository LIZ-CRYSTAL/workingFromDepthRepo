#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import output_predict, csv_inputs
import model
import train_operation as op

from tensorflow.python.client import timeline
import time

MAX_STEPS = 10000000
LOG_DEVICE_PLACEMENT = False #Whether to log device placement.
BATCH_SIZE = 8
LOG_FREQUENCY = 10
TRAIN_FILE = "train.csv"
COARSE_DIR = "coarse"
REFINE_DIR = "refine"
checkpoint_dir = './here'

REFINE_TRAIN = True
FINE_TUNE = True

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, depths, invalid_depths = csv_inputs(TRAIN_FILE, BATCH_SIZE)

        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)

        # Build graph
        coarse = model.inference(images, keep_conv, trainable=False)
        logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)
        loss = model.loss(logits, depths, invalid_depths)
        train_op = op.train(loss, global_step, BATCH_SIZE)
        
        # Logger


        class _LoggerHook(tf.train.SessionRunHook):
          """Logs loss and runtime."""

          def begin(self):
            self._step = -1
            self._start_time = time.time()

          def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(loss)  # Asks for loss value.

          def after_run(self, run_context, run_values):
            if self._step % LOG_FREQUENCY == 0:
              current_time = time.time()
              duration = current_time - self._start_time
              self._start_time = current_time

              loss_value = run_values.results
              examples_per_sec = LOG_FREQUENCY * BATCH_SIZE / duration
              sec_per_batch = float(duration / LOG_FREQUENCY)

              format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              print (format_str % (datetime.now(), self._step, loss_value,
                                   examples_per_sec, sec_per_batch))

        # Train
        with tf.train.MonitoredTrainingSession(
                save_checkpoint_secs=30,
                checkpoint_dir=REFINE_DIR,
                hooks=[tf.train.StopAtStepHook(last_step=MAX_STEPS),
                    tf.train.NanTensorHook(loss),
                    _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=LOG_DEVICE_PLACEMENT)) as mon_sess:

            while not mon_sess.should_stop():
                mon_sess.run(train_op, feed_dict={keep_conv: 0.8, keep_hidden: 0.5})



def main(argv=None):
    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()
