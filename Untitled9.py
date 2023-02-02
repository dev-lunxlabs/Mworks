#!/usr/bin/env python
# coding: utf-8

# ### Objective - To Train a model for three trips classifications 
# - Local short distance Trips 
# - Local long distance Trips 
# - Long distance Trips 
# 
# ### Data 
# Total Trips - 10,000 Trips 
# Trip Data - /Users/PK/Desktop/Mworks/training_data_unlabelled.csv

# # Data Analysis 

# We Plot the entire dataset (10,000 trips) to identify what thresholds can be used for 
# - Local short distance Trips 
# - Local long distance Trips 
# - Long distance Trips 

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/PK/Desktop/Mworks/training_data_unlabelled.csv')

#All points
total = df.shape[0]
print("Total number of points:", total)

df.plot(x='route_id', y='trip_distance', kind='scatter')
plt.show()

#Points under 200
under_200 = df[df['trip_distance'] < 200]
count = under_200.shape[0]
print("Number of points under 200:", count)


df.plot(x='route_id', y='trip_distance', kind='scatter')
plt.ylim(0, 200)
plt.show()

#Points under 25
under_25 = df[df['trip_distance'] < 25]
count = under_25.shape[0]
print("Number of points under 25:", count)


df.plot(x='route_id', y='trip_distance', kind='scatter')
plt.ylim(0, 25)
plt.show()


# The above data analysis gives us three threshold to classify trips 

# # Labelling Data 

# ### We have 3 threshold now 
#  
# We use the code below to label the data using the set 3 threshholds for trips 
# 
# - under 25 = local short distance
# - between 25 and 300 = local short distance
# - over 350 = long distance

# In[11]:


import csv

# Define the thresholds for different types of trips
local_short_distance_threshold = 25
local_long_distance_threshold = 300
long_distance_threshold = 300

local_short_distance_trips = 0
local_long_distance_trips = 0
long_distance_trips = 0

# Read the trip data from the CSV file
with open("/Users/PK/Desktop/Mworks/training_data_unlabelled.csv", "r") as f:
    reader = csv.DictReader(f)
    trips = list(reader)

# Create an empty list to store the classification of each trip
classification = []

# Iterate over the trips in your dataset
for trip in trips:
    distance = float(trip["trip_distance"])
    if distance <= local_short_distance_threshold:
        classification.append("local-short-distance")
        local_short_distance_trips += 1
    elif distance <= local_long_distance_threshold:
        classification.append("local-long-distance")
        local_long_distance_trips += 1
    elif distance > long_distance_threshold:
        classification.append("long-distance")
        long_distance_trips += 1

print("Total local-short-distance trips:", local_short_distance_trips)
print("Total local-long-distance trips:", local_long_distance_trips)
print("Total long-distance trips:", long_distance_trips)

# Write the classification results to a new CSV file
with open("/Users/PK/Desktop/Mworks/training_data_labelled_new_feb1", "w") as f:
    fieldnames = ["route_id", "classification", "distance"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, cls in enumerate(classification):
        writer.writerow({"route_id": trips[i]["route_id"], "classification": cls, "distance": trips[i]["trip_distance"]})


# # Logistic Regression Model
# 
# Now we will use a logistic Regression algorithm to first train our modle using 10% of the data set (1000 trips) 
# and then use the mode to make predection of teh rest of the 90% of teh data set (9000 trips)

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# In[13]:


# Load the training data into a Pandas dataframe
df = pd.read_csv('/Users/PK/Desktop/Mworks/training_data_labelled_new_feb1')

print(df.head())


# In[14]:


# Define the thresholds for the different classifications
local_short_distance_threshold = 25
local_long_distance_threshold = 300

# Modify the labeling of the trips based on the thresholds
df['classification'] = 'long_distance'
df.loc[(df['distance'] > 0) & (df['distance'] <= local_short_distance_threshold), 'classification'] = 'local_short_distance'
df.loc[(df['distance'] > local_short_distance_threshold) & (df['distance'] <= local_long_distance_threshold), 'classification'] = 'local_long_distance'


# In[15]:


# Split the data into training and test sets
X = df[['distance']]
y = df['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
log_reg = LogisticRegression()

# Fit the model to the training data
log_reg.fit(X_train, y_train)

# Use the model to classify the test data
y_pred = log_reg.predict(X_test)


# In[16]:


# Evaluate the performance of the model
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
print("Accuracy: ", acc)
print("Confusion Matrix: \n", conf_mat)


# Now we make predections on the 90% of the unlabelled data

# In[17]:


# load the unlabelled data
unlabelled_data = pd.read_csv('/Users/PK/Desktop/Mworks/training_data_unlabelled2.csv')

# extract the distance column
unlabelled_distance = unlabelled_data[['distance']]

# use the trained model to classify the unlabelled data
unlabelled_labels = log_reg.predict(unlabelled_distance)

# add labels to the unlabelled dataframe
unlabelled_data['label'] = unlabelled_labels

# save the unlabelled data with labels
unlabelled_data.to_csv("/Users/PK/Desktop/Mworks/predection_result_unlabelled_data_with_label.csv", index=False)



# ### Conclusions form the LR Model 
# 
# * Accuracy - 100%
# * Training Data Used - 1000 Trips 
# * Un lablelled Predection data - 9000 Trips
# 
# 
# ### Predection on the Un-labelled Data 
# 
# *Classification Values* 
# * Local Short Distance 
# * Local Long Distance 
# * Long Distance 

# In[18]:


df = pd.read_csv('/Users/PK/Desktop/Mworks/predection_result_unlabelled_data_with_label.csv')

print(df.head())

