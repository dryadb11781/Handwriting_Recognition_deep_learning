from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
from os.path import join,basename
import sys
print('train_file_path:',os.path.split(os.path.realpath(__file__))[0])
base_path=os.path.split(os.path.realpath(__file__))[0]
csv_path=join(base_path,'data')
tf_model_path=join(base_path,'model_save')

#output_dir = sys.argv[1]
tf.logging.set_verbosity(tf.logging.INFO)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)


  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)


  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)


  pool3_flat = tf.reshape(pool3, [-1, 8 *8 * 64])

  dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)


  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)


  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def csv_data_read():
    import glob
    import csv
    import numpy as np
    #from matplotlib import pyplot as plt
    files=glob.glob(csv_path+'\*.csv')
    print('totaly csv:',len(files))
    n = 0
    data = []
    for temp in files:
        f = open('%s'%temp, 'r') #000.csv 改為自己的手寫辨識資料檔名
        for row in csv.reader(f):
            data.append(row)
        f.close()


    for n in range(len(data)):
        for i in range(len(data[0])):
            data[n][i] = int(data[n][i])

    image = []
    target = []

    for n in range(len(data)):
        t = data[n][0]
        target.append(t)
        image.append(data[n][1:])

    image = np.array(image)

    return (image,target)


def main(unused_argv):
  # Load training and eval data
  image,target = csv_data_read()
  train_data = np.float32(image)  # Returns np.array
  target=np.float32(target)
  print('train_data',train_data.shape)
  train_labels = np.asarray(target, dtype=np.int32)
  print('train_labels',train_labels.shape)
  eval_data = np.float32(image)  # Returns np.array
  eval_labels = np.asarray(target, dtype=np.int32)


#float32
#  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#  train_data = mnist.train.images  # Returns np.array
#  #print('################',type(train_data[0,0]))
#  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#  eval_data = mnist.test.images  # Returns np.array
#  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=tf_model_path)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=2000)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=19000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
