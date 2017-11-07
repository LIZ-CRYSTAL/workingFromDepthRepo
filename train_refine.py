#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
import model
import train_operation as op
from PIL import Image


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

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

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
        
        tf.summary.image('images2', logits*255.0, max_outputs=3)
        
        loss = model.loss(logits, depths, invalid_depths)
        train_op = op.train(loss, global_step, BATCH_SIZE)
        
        #load the values from the coarse network
        #FIXME: These files will be overriden by the MonitoredTrainingSession, which will cause problems and is the wrong way of doing this!!
        saver_coarse = tf.train.Saver(coarse.all_variables())
        # Logger
        class _LoggerHook(tf.train.SessionRunHook):
          """Logs loss and runtime."""

          def begin(self):
            self._step = -1
            self._start_time = time.time()

          def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs([loss, logits, images])  # Asks for loss value.

          def after_run(self, run_context, run_values):
            if self._step % LOG_FREQUENCY == 0:
              current_time = time.time()
              duration = current_time - self._start_time
              self._start_time = current_time

              loss_value = run_values.results[0]
              depths = run_values.results[1]
              images = run_values.results[2]
              output_predict(depths, images, 'refine');
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

def csv_inputs(csv_file_path, batch_size):
    filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
    reader = tf.TextLineReader()
    _, serialized_example = reader.read(filename_queue)
    filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
    # input
    jpg = tf.read_file(filename)
    image = tf.image.decode_jpeg(jpg, channels=3)
    image = tf.cast(image, tf.float32)
    # target
    depth_png = tf.read_file(depth_filename)
    depth = tf.image.decode_png(depth_png, channels=1)
    depth = tf.cast(depth, tf.float32)
    depth = tf.div(depth, [255.0])
    #depth = tf.cast(depth, tf.int64)
    # resize
    image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
    invalid_depth = tf.sign(depth)
    # generate batch
    images, depths, invalid_depths = tf.train.batch(
        [image, depth, invalid_depth],
        batch_size=batch_size,
        num_threads=4,
        capacity= 50 + 3 * batch_size,
    )
    tf.summary.image('images', images, max_outputs=3)
    return images, depths, invalid_depths

def output_predict(depths, images, output_dir):
    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth) in enumerate(zip(images, depths)):
        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%05d_org.png" % (output_dir, i)
        pilimg.save(image_name)
        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d.png" % (output_dir, i)
        depth_pil.save(depth_name)


def main(argv=None):
    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()
