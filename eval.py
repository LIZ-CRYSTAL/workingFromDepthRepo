eval_dir = './here'
eval_interval_secs = 600
import math
import model
import task
import tensorflow as tf
import numpy as np
import datetime
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


num_examples = 100 #FIXME: WHAT DOES THIS DO ?????

def eval_once(saverCoarse, saverRefine, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  config = tf.ConfigProto(device_count = {'GPU': 0})
  with tf.Session(config=config) as sess:
    ckptCoarse = tf.train.get_checkpoint_state('./coarse')
    ckptRefine = tf.train.get_checkpoint_state('./refine')
    if ckptCoarse and ckptCoarse.model_checkpoint_path and ckptRefine and ckptRefine.model_checkpoint_path :
      # Restores from checkpoint
      saverCoarse.restore(sess, ckptCoarse.model_checkpoint_path)
      saverRefine.restore(sess, ckptRefine.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckptCoarse.model_checkpoint_path.split('/')[-1].split('-')[-1]
      global_step2 = ckptRefine.model_checkpoint_path.split('/')[-1].split('-')[-1]
      if not global_step2 == global_step:
          print('DIDNT MATCH !!!!!!!!')
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(num_examples / 8))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter *8 
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (100.0, precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

from dataset import DataSet
import time
def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    dataset = DataSet(8)
    images, depths, invalid_depths = dataset.csv_inputs('test.csv')
    # Build a Graph that computes the logits predictions from the
    # inference model.
    keep_conv = tf.placeholder(tf.float32)
    keep_hidden = tf.placeholder(tf.float32)
    coarse = model.inference(images, trainable=False)
    logits = model.inference_refine(images, coarse, .5, keep_hidden)
    # Calculate predictions.
    top_k_op = model.loss(logits, depths, invalid_depths)
    
    init_op = tf.global_variables_initializer()
    
    # Session
    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(init_op)
    
        coarse_params = {}
        refine_params = {}
                    #clean this up
        for variable in tf.global_variables():
            variable_name = variable.name
            if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                continue
            if variable_name.find('coarse') >= 0:
                coarse_params[variable_name] = variable
            if variable_name.find('fine') >= 0:
                refine_params[variable_name] = variable
        # Restore the moving average version of the learned variables for eval.
        #variable_averages = tf.train.ExponentialMovingAverage(
        #    MOVING_AVERAGE_DECAY)
        #variables_to_restore = variable_averages.variables_to_restore()
        saverCoarse = tf.train.Saver(coarse_params)
        saverRefine = tf.train.Saver(refine_params)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(eval_dir, g)

        while True:
          eval_once(saverCoarse, saverRefine, summary_writer, top_k_op, summary_op)
          time.sleep(eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(eval_dir):
    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)
  with tf.device('/cpu:0'):
    evaluate()


if __name__ == '__main__':
  tf.app.run()

