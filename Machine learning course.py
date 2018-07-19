# -*- coding: utf-8 -*-
import tensorflow as tf
hello = tf.constant("Hello, World!")
sess = tf.Session()
print (sess.run(hello))

##########Linear regression

#Set up a linear classifier
#classifier = tf.estimator.LinearClassifier()

#Train model on some example data 
#classifier.train (input_fn = train_input_fn, steps = 2000)

#Use it to predict
#predictions = classifier.predict (input_fn = predict_input_fn)

##########Learning to use pandas package
##For using pandas library
import pandas as pd
pd.__version__
pd.Series (['San Francisco', 'San Jose', 'Sacramento'])

city_names = pd.Series(['San Francisco', "San Jose", "Sacramento"])
population = pd.Series([852469, 1015785, 485199])
pd.DataFrame({'City name': city_names, 'Population': population})

#########Loading csv from URL
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()
california_housing_dataframe.head()
california_housing_dataframe.hist('latitude')

#########Creating a new dataframe form given data above
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
cities['City name']
cities ['City name'][1] ##For accessing to an array value
cities[0:2] ##For accessing multiple array values


##########Math operations with the given data
population / 1000
##For using NumPy library
import numpy as np
np.log(population)
