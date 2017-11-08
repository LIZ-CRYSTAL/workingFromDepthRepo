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
        global_step = tf.Variable(0, trainable=False)
        images, depths, invalid_depths = csv_inputs(TRAIN_FILE, BATCH_SIZE)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        
        print("refine train.")
        coarse = model.inference_coarse(images, keep_conv, trainable=False)
        logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)
        
        loss = model.loss(logits, depths, invalid_depths)
        train_op = op.train(loss, global_step, BATCH_SIZE)
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init_op)    

        # parameters
        coarse_params = {}
        variableName = "inference_coarse"
        varsToLoad = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, variableName)
        # define saver
        saver_coarse = tf.train.Saver(varsToLoad)
        # fine tune
        coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
        if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
            print("Pretrained coarse Model Loading.")
            saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
            print("Pretrained coarse Model Restored.")
        else:
            raise

        refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
        if refine_ckpt and refine_ckpt.model_checkpoint_path:
            print("Pretrained refine Model Loading.")
            #saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
            print("Pretrained refine Model Restored.")
        else:
            print("No Pretrained refine Model.")

        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in xrange(MAX_STEPS):
            index = 0
            for i in xrange(1000):
                _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                if index % 10 == 0:
                    print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if index % 500 == 0:
                    output_predict(logits_val, images_val, "data/predict_refine_%05d_%05d" % (step, i))
                index += 1

            if step % 5 == 0 or (step * 1) == MAX_STEPS:
                refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                saver_refine.save(sess, refine_checkpoint_path, global_step=step)
        coord.request_stop()
        coord.join(threads)
        sess.close()

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
