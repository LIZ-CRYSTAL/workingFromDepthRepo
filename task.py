#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict
import model
import train_operation as op

from tensorflow.python.client import timeline


MAX_STEPS = 10000000
LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 8
TRAIN_FILE = "train.csv"
COARSE_DIR = "coarse"
REFINE_DIR = "refine"
checkpoint_dir = './here'

REFINE_TRAIN = True
FINE_TUNE = True

def train():
    with tf.Graph().as_default():
        # Session
        with tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT,allow_soft_placement = True)) as sess:
            global_step = tf.Variable(0, trainable=False)
            dataset = DataSet(BATCH_SIZE)
            images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
            keep_conv = tf.placeholder(tf.float32)
            keep_hidden = tf.placeholder(tf.float32)

            coarse = model.inference(images, keep_conv, trainable=False)
            logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)

            loss = model.loss(logits, depths, invalid_depths)
            train_op = op.train(loss, global_step, BATCH_SIZE)
            init_op = tf.global_variables_initializer()

            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            print 'starting init op'
            sess.run(init_op)
            print '/starting init op'

            
            writer = tf.summary.FileWriter("/tmp/tensorflow/depth1", sess.graph)
            merged = tf.summary.merge_all()
            # parameters
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
            
            # define saver
            print "Saving coarse_params and refine_params"
            saver_coarse = tf.train.Saver(coarse_params)
            saver_refine = tf.train.Saver(refine_params)

            #restore the models
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")

            refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
            if refine_ckpt and refine_ckpt.model_checkpoint_path:
                print("Pretrained refine Model Loading.")
                saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                print("Pretrained refine Model Restored.")
            else:
                print("No Pretrained refine Model.")

            # train
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for step in xrange(MAX_STEPS):
                index = 0
                for i in xrange(1000):
                    print 'starting run'
                    if index % 10 == 0:
                    	_, loss_value, logits_val, images_val, summ  = sess.run([train_op, loss, logits, images, merged], feed_dict={keep_conv: 0.8, keep_hidden: 0.5}, options=options, run_metadata=run_metadata)
                    else:
                    	_, loss_value, logits_val, images_val  = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5}, options=options, run_metadata=run_metadata)
                    print '/starting run'

                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('timeline_01.json', 'w') as f:
                        f.write(chrome_trace)

                    if index % 10 == 0:
                        writer.add_summary(summ)
                        print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))
                        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    if index % 500 == 0:
                        if REFINE_TRAIN:
                            output_predict(logits_val, images_val, "data/predict_refine_%05d_%05d" % (step, i))
                        else:
                            output_predict(logits_val, images_val, "data/predict_%05d_%05d" % (step, i))
                    index += 1

                if step % 5 == 0 or (step * 1) == MAX_STEPS:
                    print 'checkpoint saved'
                    #FIXME: This used to only save one (in our case the 
                    refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path, global_step=step)
                    coarse_checkpoint_path = COARSE_DIR + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)



def main(argv=None):
    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()
