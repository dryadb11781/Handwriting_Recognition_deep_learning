from django.http import HttpResponse
from django.template.loader import get_template
from django import template
from django.shortcuts import render_to_response
from django.views.decorators.csrf import csrf_exempt
import os

def index(request):
    return render_to_response('index.html',locals())

def data_generator(request):
    return render_to_response('hand_number_data_generator.html',locals())


def upload_csv_data(request):
    csv_data=request.GET['csv_data']
    print("I got data")
    return HttpResponse("Thank you")
@csrf_exempt
def upload_csv_data_post(request):
    import os
    from os.path import join,basename
    import sys
    from time import gmtime, strftime
    base_path=os.path.split(os.path.realpath(__file__))[0]
    print('base_path:',base_path)
    data_save_dir=join(base_path,'..','..','data')
    print('data_save_dir:',data_save_dir)
    csv_data=request.POST['csv_data']
    user_name=request.POST['user_name']
    data_save_dir=join(base_path,'..','..','data',user_name)
    now_time=str(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    with open(data_save_dir+now_time+'.csv', 'a') as the_file:
        writestr = csv_data
        the_file.write(writestr)
    return HttpResponse("Thank you")







def predict(request):
    import numpy as np
    data = np.array(request.GET['test'].split(','))
    X_test=np.zeros((1,4096))
    X_test[0,:]=data
    print('X_TEST',X_test.shape)
    import numpy as np
    import tensorflow as tf
    import os
    from os.path import join,basename
    import sys
    base_path=os.path.split(os.path.realpath(__file__))[0]

    tf_model_path=join(base_path,'..\..','model_save')

    tf.logging.set_verbosity(tf.logging.INFO)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
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

    def tf_predict_number(image):
      # Load training and eval data
      eval_data = np.float32(image)  # Returns np.array
    #  eval_labels = np.asarray(target, dtype=np.int32)



      mnist_classifier = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir=r"D:\hand_reconiciton\models-master\models-master\tutorials\image\mnist\model_save")

      # Set up logging for predictions
      # Log the values in the "Softmax" tensor with label "probabilities"
    #  tensors_to_log = {"probabilities": "softmax_tensor"}

      predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        num_epochs=1,
        shuffle=False)
      predictions = mnist_classifier.predict(input_fn=predict_input_fn)
      for i,result in enumerate(predictions):
          print("Prediction" ,result["classes"])
      return result["classes"]



    return HttpResponse(tf_predict_number(X_test))
# Create your views here.
