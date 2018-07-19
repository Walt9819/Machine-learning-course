# -*- coding: utf-8 -*-
import tensorflow as tf
hello = tf.constant("Hello, World!")
sess = tf.Session()
print (sess.run(hello))

##########Linear regression

#Set up a linear classifier
classifier = tf.eestimator.LinearClassifier()

#Train model on some example data 
classifier.train (input_fn = train_input_fn, steps = 2000)

#Use it to predict
predictions = classifier.predict (input_fn = predict_input_fn)

##########Lear to using pandas package
