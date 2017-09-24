import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import math
import csv
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
#####################
path_train = "doc2vec.csv"

path_scores="doc2vecIndex.csv"
threshold=50
print("Loading Train Data")
train = pd.read_csv(path_train, sep=",")
print train.count
#train=train.iloc[:,:20]

scores=pd.read_csv(path_scores, sep=",")
print scores.count
print scores['score']
scores_list=scores['score']
scores_list.astype(float);
targets=[]
for idx in xrange(len(scores_list)):
    #if scores_list[idx]>threshold:
    #    targets.append(1)
    #else:
        #targets.append(0)
    #print max(scores_list[idx],0.01),scores_list[idx],math.log(max(scores_list[idx],0.01))
    targets.append(math.log(max(scores_list[idx],0.0001)))
targets = pd.DataFrame(targets)
print targets
########################
def train_nn_regression_model(
    optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """
  In addition to training, this function also prints training progress information,
  a plot of the training and validation loss over time, as well as a confusion
  matrix.

  Args:
    learning_rate: An `int`, the learning rate to use.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
    training_examples: A `DataFrame` containing the training features.
    training_targets: A `DataFrame` containing the training labels.
    validation_examples: A `DataFrame` containing the validation features.
    validation_targets: A `DataFrame` containing the validation labels.

  Returns:
    A tuple `(estimator, training_losses, validation_losses):
      estimator: the trained `DNNRegressor` object.
      training_losses: a `list` containing the training loss values taken during training.
      validation_losses: a `list` containing the validation loss values taken during training.
  """

  periods = 10
  steps_per_period = steps / periods

  # Create the input functions.
  feature_columns = set([tf.contrib.layers.real_valued_column(my_feature) for my_feature in training_examples])

  training_input_fn = learn_io.pandas_input_fn(
     x=training_examples, y=training_targets,
     num_epochs=None, batch_size=batch_size)

  predict_training_input_fn = learn_io.pandas_input_fn(
     x=training_examples, y=training_targets,
     num_epochs=1, shuffle=False)

  predict_validation_input_fn = learn_io.pandas_input_fn(
       x=validation_examples, y=validation_targets,
      num_epochs=1, shuffle=False)


  # Create a linear regressor object.

  dnn_regressor = tf.contrib.learn.DNNRegressor(
      feature_columns=feature_columns,
      hidden_units=hidden_units,
      optimizer=optimizer,
      gradient_clip_norm=5.0
  )

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print "Training model..."
  print "RMSE (on training data):"
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.fit(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = list(dnn_regressor.predict(input_fn=predict_training_input_fn))
    validation_predictions = list(dnn_regressor.predict(input_fn=predict_validation_input_fn))
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print "  period %02d : %0.2f" % (period, training_root_mean_squared_error)
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print "Model training finished."

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  print "Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error
  print "Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error

  return dnn_regressor, training_rmse, validation_rmse


###############

nb_folds = 4
kfolds = KFold(len(targets), nb_folds, shuffle=True, random_state=1337)
av_roc = 0.
f = 0

print kfolds


#    To train the DNN classifier with features and target data
#    :param features:
#    :param target:
#    :return: trained  classifier

for trainrows, validrows in kfolds:
    print('---'*20)
    print('Fold', f)
    print('---'*20)
    f += 1
    train_rows = train.iloc[trainrows,:]
    train_validation = train.iloc[validrows,:]
    labels_train = targets.iloc[trainrows]
    labels_validation = targets.iloc[validrows]

    _, adagrad_training_losses, adagrad_validation_losses = train_nn_regression_model(
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.5),
    steps=10000,
    batch_size=20,
    hidden_units=[1024, 512, 256,128],
    training_examples=train_rows,
    training_targets=labels_train,
    validation_examples=train_validation,
    validation_targets=labels_validation)

    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.plot(adagrad_training_losses, label='Adagrad training')
    plt.plot(adagrad_validation_losses, label='Adagrad validation')
    _ = plt.legend()
    plt.show()
