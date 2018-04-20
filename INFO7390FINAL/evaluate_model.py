import tensorflow as tf
import numpy as np
import os
import time
import datetime
import cnn_model
from cnn_model import TextCNN
from tensorflow.contrib import learn
import csv
import sys

# Parameters
# ==================================================


# find the lastest folder for model

folderList = []


for dirs in os.listdir('runs/'):
    dirstring = 'runs/' + dirs
    if os.path.isdir(dirstring):
        folderList.append(dirstring)

folderString = folderList[len(folderList) - 1]
folderString = folderString + '/'
checkpointString = folderString + '/checkpoints' 
print(checkpointString)

# Data Parameters
tf.flags.DEFINE_string("path", "./pre/", "Data source.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", folderString, "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_string("testdata_path", checkpointString, "checkpoint directory")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]
else:
    #x_raw, y_test = cnn_model.load_data_and_labels5(FLAGS.path)
	#if you are using load_data_and_labels3 in train_model, use it here as well
    x_raw, y_test = cnn_model.load_data_and_labels3(FLAGS.path)
    y_test = np.argmax(y_test, axis=1)
	
    

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
print(vocab_path)
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.testdata_path)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = cnn_model.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv


# decode y_test and all_predictions
text_label = []
for i in range(len(y_test)):
    if y_test[i] == 0:
        text_label.append('steam')
    if y_test[i] == 1:
        text_label.append('nytimes')
    if y_test[i] == 2:
        text_label.append('NASA')
    if y_test[i] == 3:
        text_label.append('linkedin')
    if y_test[i] == 4:
        text_label.append('bbcnews')

prediction_label = []
for i in range(len(all_predictions)):
    if all_predictions[i] == 0:
        prediction_label.append('steam')
    if all_predictions[i] == 1:
        prediction_label.append('nytimes')
    if all_predictions[i] == 2:
        prediction_label.append('NASA')
    if all_predictions[i] == 3:
        prediction_label.append('linkedin')
    if all_predictions[i] == 4:
        prediction_label.append('bbcnews')

predictions_human_readable = np.column_stack((np.array(x_raw), prediction_label, text_label))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
