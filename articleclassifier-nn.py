import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import csv
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

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
train=train.iloc[:,:20]

scores=pd.read_csv(path_scores, sep=",")
print scores.count
print scores['score']
scores_list=scores['score']
targets=[]
for idx in xrange(len(scores_list)):
    if scores_list[idx]>threshold:
        targets.append(1)
    else:
        targets.append(0)
targets = pd.DataFrame(targets)
print targets
########################
def train_nn_classification_model(
    learning_rate,
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
    The trained `DNNClassifier` object.
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


  # Create a linear classifier object.

  classifier = tf.contrib.learn.DNNClassifier(
      feature_columns=feature_columns,
      n_classes=2,
      hidden_units=hidden_units,
      optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate),
      gradient_clip_norm=5.0,
      config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1))

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.

  print "Training model..."
  print "Per period ROC (on training data):"
  #training_errors = []
  #validation_errors = []

  training_loglossvals = []
  validation_loglossvals = []
  training_rocvals = []
  validation_rocvals = []

  #training_precisionvals = []
  #validation_precisionvals = []
  #training_recallvals = []
  #validation_recallvals = []
  #thresh=0.7
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    classifier.fit(
        input_fn=training_input_fn,
        steps=steps_per_period)

    # Take a break and compute predictions.
    training_predictions = np.array(list(classifier.predict_proba(input_fn=predict_training_input_fn)))
    validation_predictions = np.array(list(classifier.predict_proba(input_fn=predict_validation_input_fn)))
    # Compute training and validation errors.
    training_log_loss = metrics.log_loss(training_targets, training_predictions[:,1])
    validation_log_loss = metrics.log_loss(validation_targets, validation_predictions[:,1])
    training_roc = metrics.roc_auc_score(training_targets, training_predictions[:, 1])
    validation_roc = metrics.roc_auc_score(validation_targets, validation_predictions[:, 1])
    #training_precision = metrics.precision_score(training_targets, (training_predictions[:, 1]>thresh).astype(int))
    #validation_precision = metrics.precision_score(validation_targets, validation_predictions[:, 1])
    #training_recall = metrics.recall_score(training_targets, training_predictions[:, 1])
    #validation_recall = metrics.recall_score(validation_targets, validation_predictions[:, 1])
    # Occasionally print the current loss.
    print "  period %02d : %0.2f" % (period, training_roc)
    # Add the loss metrics from this period to our list.
    training_rocvals.append(training_roc)
    validation_rocvals.append(validation_roc)
    training_loglossvals.append(training_log_loss)
    validation_loglossvals.append(validation_log_loss)

    #training_precisionvals.append(training_precision)
    #validation_precisionvals.append(validation_precision)
    #training_recallvals.append(training_recall)
    #validation_recallvals.append(validation_recall)


  print "LogLoss error (on training data):"
  print training_loglossvals
  print "LogLoss error (on validation data):"
  print validation_loglossvals
  print "ROC (on training data):"
  print training_rocvals
  print "ROC (on validation data):"
  print validation_rocvals
  #print "Precision (on training data):"
  #print training_precisionvals
  #print "Precision (on validation data):"
  #print validation_precisionvals
  #print "Recall (on training data):"
  #print training_recallvals
  #print "Recall (on validation data):"
  #print validation_recallvals
  print "Model training finished."
  return classifier


  # Remove event files to save disk space.
  #_ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

  # Calculate final predictions (not probabilities, as above).
  #final_predictions = list(classifier.predict(validation_examples))
  #accuracy = metrics.accuracy_score(validation_targets, final_predictions)
  #print "Final accuracy (on validation data): %0.2f" % accuracy

  # Output a graph of loss metrics over periods.
  #plt.ylabel("LogLoss")
  #plt.xlabel("Periods")
  #plt.title("LogLoss vs. Periods")
  #plt.plot(training_errors, label="training")
  #plt.plot(validation_errors, label="validation")
  #plt.legend()
  #plt.show()

  # Output a plot of the confusion matrix.
  #cm = metrics.confusion_matrix(validation_targets, final_predictions)
  # Normalize the confusion matrix by row (i.e by the number of samples
  # in each class)
  #cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  #ax = sns.heatmap(cm_normalized, cmap="bone_r")
  #ax.set_aspect(1)
  #plt.title("Confusion matrix")
  #plt.ylabel("True label")
  #plt.xlabel("Predicted label")
  #plt.show()


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

    classifier = train_nn_classification_model(
    learning_rate=0.005,
    steps=10000,
    batch_size=20,
    hidden_units=[1024, 512, 256],
    training_examples=train_rows,
    training_targets=labels_train,
    validation_examples=train_validation,
    validation_targets=labels_validation)