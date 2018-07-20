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

##Select from dataframes the ones with specific criteria
population.apply(lambda val : val > 1000000)

##Adding columns to dataframes
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92]) 
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities


###########First excersice (Create two new boolean columns, one if the 
##city has "San" and other if the square area is greater than 50 miles)

##Has "San" on city name
cities['San on city'] = cities['City name'].apply(lambda name: name.startswith("San")) 
cities
#Square area greater than 50 miles
cities['Area greater'] = cities ['Area square miles'].apply(lambda val : val > 50) 
cities

#If combine both instructions use "&" instead of "and"
cities['Is wide and have San on name'] = cities['City name'].apply(lambda name: name.startswith("San")) & cities ['Area square miles'].apply(lambda val : val > 50)
cities


##########Index with pandas
city_names.index
cities.index

##Give manually an index place to the given data
cities.reindex([2, 0, 1])
##Reindex with random from numpy
cities.reindex(np.random.permutation(cities.index))


##########Second excercise (Try to give index values that wasn´t on the original index values)
cities.reindex([4, 2, 8])
