from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from imblearn.pipeline import make_pipeline
import tensorflow as tf
from sklearn.utils import class_weight
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
matplotlib.use('TkAgg')
from keras.layers import Dropout
from keras import regularizers
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from scipy import stats
from keras.models import model_from_json
import numpy
from keras.models import Sequential
from keras.layers import Dense
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import pickle
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pandas.api.types import is_string_dtype


import time
import django
print(django.get_version())

rcParams['figure.figsize']=10,6
warnings.filterwarnings('ignore')
sns.set(style="darkgrid")
df=pd.read_csv("/home/rifaza/Final_Year_Project/Spartans/Transportation/CO.csv")
def readAllData():
    df=pd.read_csv("/home/rifaza/Final_Year_Project/Spartans/Transportation/CO.csv")
    print("Shape of the data :  ", df.shape)
    print("Sample data set : ")
    print(df[0:15])

readAllData()
aqiArray1=[]
print ("\n###############Actual Output####################\n")
def findAQIForTest(test_data,aqiArray1):
    for row in test_data.itertuples():
        new_AQI = (0.084 * row.NO2) + (0.052 * row.O3) + (0.047 * row.PM25)
        aqiArray1.append(round(new_AQI, 2))

    time.sleep(3)
    print(aqiArray1[0:1])
    print(type(aqiArray1))
    print("##############")

findAQIForTest(df,aqiArray1 )
#using best check point we are going to test the data
def make_submission(prediction, sub_name):

  my_submission = pd.DataFrame({'Id':pd.read_csv('CO.csv').Id,
                                'Region':pd.read_csv('CO.csv').Region,
                                'Time_GMT':pd.read_csv('CO.csv').Time_GMT,
                                'Wday': pd.read_csv('CO.csv').Wday,
                                'O3': pd.read_csv('CO.csv').O3,
                                'PM25': pd.read_csv('CO.csv').PM25,
                                'NO2':pd.read_csv('CO.csv').NO2,
                                'Place_ID': pd.read_csv('CO.csv').Place_ID,

                                'AQI':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')


###here we are saving the output ##########
make_submission(aqiArray1,'New_Final_File')
df.insert(7, "AQI", aqiArray1, True)
print("\nchecking the outliers- if the graph  will be displayed if there is only the outliers\n")
sns.boxplot(x=df['NO2'])

sns.boxplot(x=df['O3'])

sns.boxplot(x=df['PM25'])

print("Null value available or not ?? " , df.isnull().any().any())
#Removed the null value if it is
df = df.dropna(inplace=False)
print("Shape of the train data" , df.shape)

print("Describetion of the dataset: ")
print(df.describe())

C_mat = df.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()


def get_data():
    train=df.drop('Place_ID', axis=1)
    test=pd.read_csv('/home/rifaza/Final_Year_Project/Spartans/Transportation/Final_Test.csv')
    test=test.drop('Place_ID', axis=1)
    z = np.abs(stats.zscore(train))
    threshold = 4
    print(np.where(z >=4))
    print(z)
    print(z[1602][4])
    train = train[(z< 4).all(axis=1)]

    print(train[0:10])
    print("\n After removing the outliers the shape of the data",train.shape)
    print("\n")
    z = np.abs(stats.zscore(test))
    threshold = 4
    print(np.where(z >=4))
    sns.boxplot(x=test['PM25'])
    # test = test[(z < 4).all(axis=1)]
    print("test Shape\n",test.shape)
    return train, test

def get_combined_data():

  train , test = get_data()

  target = train.AQI
  train.drop(['AQI'],axis = 1 , inplace = True)

  combined = train.append(test)
  combined.reset_index(inplace=True)
  combined.drop(['index', 'Id'], inplace=True, axis=1)

  scaler = MinMaxScaler()
  print(scaler.fit(combined))
  MinMaxScaler()

  print(scaler.transform(combined))

  return combined, target


#Load train and test data into pandas DataFrames
train_data, test_data = get_data()

print("AQI vs NO2 concentration")
fig, ax = plt.subplots()
ax.scatter(x = train_data['NO2'], y = train_data['AQI'])
plt.ylabel('AQI', fontsize=13)
plt.xlabel('NO2 Concentration', fontsize=13)
plt.show()


#Combine train and test data to process them together
combined, target = get_combined_data()
#This function used to get the colun that dont have null values
def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type :
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans

num_cols = get_cols_with_no_nans(combined , 'num')
cat_cols = get_cols_with_no_nans(combined , 'no_num')

print ('Number of numerical columns with no nan values :',len(num_cols))
print ('Number of nun-numerical columns with no nan values :',len(cat_cols))


train_data = train_data[num_cols + cat_cols]
train_data['Target'] = target



def split_combined():
    global combined
    print(combined.shape)
    train = combined[:5369]
    test = combined[5369:]

    return train, test


train, test = split_combined()

NN_model = Sequential()


# The Input Layer :
NN_model.add(Dense(200, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :

NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
NN_model.summary()
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

NN_model.fit(train, target, epochs=100, batch_size=32, validation_split = 0.25, callbacks=callbacks_list)
print("\nLoad wights file of the best model :\n")

wights_file = 'Weights-099--0.00001.hdf5' # choose the best checkpoint
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

#using best check point we are going to test the data
def make_submission(prediction, sub_name):

  my_submission = pd.DataFrame({'Id':pd.read_csv('Final_Test.csv').Id,'AQI':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

predictions = NN_model.predict(test)

###here we are saving the output ##########
make_submission(predictions[:,0],'Final_ANN_Prediction')
print("###########################")


print("################# Saving the model #############")

import pickle
from sklearn.externals import joblib
filename= 'trainmodel.pkl'
joblib.dump( NN_model, filename)


##########################################
output = pd.read_csv('Final_ANN_Prediction.csv')
test_data=pd.read_csv('Final_Test.csv')
print("New shape",test_data.shape)
level_of_concern = []

test_data.insert(7,"AQI",predictions,True)


for row in test_data.itertuples():

    if (row.Region == 1):
        if (row.Time_GMT >= 8) & ( row.Time_GMT <= 14) & (row.Wday == 1):

            if(row.AQI <= 10):
                level = "Good"
                l_code = 0
            elif (row.AQI <= 15):
                level = "Moderate"
                l_code = 1
            elif (row.AQI <=19):
                level = "Unhealthy"
                l_code = 2
            elif (row.AQI>19):
                level = " Very Unhealthy"
                l_code = 3

        else:
            if (row.AQI<= 16):
                level = "Good"
                l_code = 0
            elif (row.AQI<= 21):
                l_code = 1
                level = "Moderate"
            elif (row.AQI<=25):
                l_code = 2
                level = "Unhealthy"
            else:
                l_code=3
                level="Very unhealthy"


    elif (row.Region == 2):
        if (row.AQI <= 9):
            level = "Good"
            l_code = 0
        elif (row.AQI <= 14):
            level = "Moderate"
            l_code = 1
        elif (row.AQI<=18):
            level = "Unhealthy"
            l_code = 2
        else:
            level="Very unhealthy"
            l_code=3

    elif (row.Region==3):
        if (row.AQI<= 20):
            level = "Good"
            level = "Good"
            l_code = 0
        elif (row.AQI<= 28):
            level = "Moderate"
            l_code = 1
        elif (row.AQI <= 32):
            level = "Unhealthy"
            l_code = 2
        else:
            level="Very unhealthy"
            l_code=3

    level_of_concern.append(l_code)

print(level_of_concern)
test_data.insert(8, "HealthConcern", level_of_concern, True)
print(test_data)

aqiArray1 = []

print ("\n###############Actual Output####################\n")
def findAQIForTest(test_data,aqiArray1):
    for row in test_data.itertuples():
        new_AQI = (0.084 * row.NO2) + (0.052 * row.O3) + (0.047 * row.PM25)
        aqiArray1.append(round(new_AQI, 2))

    time.sleep(3)
    print(aqiArray1[0:1])
    print(type(aqiArray1))
    print("##############")

findAQIForTest(test_data,aqiArray1 )
print("\n\n Actual output for the test data\n\n",aqiArray1)
y_actual=aqiArray1
y_predicted= output['AQI']


print("\n################Random Forest Regressor###################################\n")

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.25, random_state = 14)

print('Training Features Shape:', train_X.shape)
print('Training Labels Shape:', train_y.shape)
print('Testing Features Shape:',val_X.shape)
print('Testing Labels Shape:', val_X.shape)

model = RandomForestRegressor(n_estimators =500, random_state = 42)
model.fit(train_X,train_y)

# Get the mean absolute error on the validation data
predicted_AQHI= model.predict(val_X)
MAE = mean_absolute_error(val_y , predicted_AQHI)
MSE=mean_squared_error(val_y, predicted_AQHI)
print('Random forest validation MAE = ', MAE)
print('Random forest validation MSE ==',MSE)
print("\nPredicted AQHI\n", predicted_AQHI)
aqiArrayNew = []
# findAQIForTest(val_y,aqiArrayNew )
# print(" predicted AQHI from Equation\n",aqiArrayNew)
mape = 100 * (MSE/ val_y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# Pull out one tree from the forest
tree =model.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = model.estimators_[5]
# Saving feature names
feature_list = list(train.columns)
# Export the image to a dot file

export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')
print(test)
predicted_aqhi = model.predict(test)
make_submission(predicted_aqhi,'Submission(RF).csv')
print("\n\nThe forest include only 10 trees and each tree included only the 3 level\n")
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_X, train_y)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');

# Get numerical feature importances
importances = list(model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

print("\n\n\n ***********second model************\n\n");

test_data_2=pd.read_csv('Final_Test.csv')

test_data_2=test_data_2.drop('Place_ID', axis=1)
test_data_2=test_data_2.drop('Id', axis=1)
test_data_2.insert(6,"AQI",aqiArray1,True)
new_dataframe=pd.read_csv('New_Final_File.csv')
new_dataframe=new_dataframe.drop('Id', axis=1)
new_dataframe=new_dataframe.drop('Place_ID', axis=1)
level_of_concern_new = []
level_of_concern_actual=[]
print(new_dataframe[0:10])

for row in new_dataframe.itertuples():

    if (row.Region == 1):
        if (row.Time_GMT >= 8) & ( row.Time_GMT <= 14) & (row.Wday == 1):

            if(row.AQI <= 10):
                level = "Good"
                l_code = 0
            elif (row.AQI <= 15):
                level = "Moderate"
                l_code = 1
            elif (row.AQI <=19):
                level = "Unhealthy"
                l_code = 2
            elif (row.AQI>19):
                level = " Very Unhealthy"
                l_code = 3

        else:
            if (row.AQI<= 16):
                level = "Good"
                l_code = 0
            elif (row.AQI<= 21):
                l_code = 1
                level = "Moderate"
            elif (row.AQI<=25):
                l_code = 2
                level = "Unhealthy"
            else:
                l_code=3
                level="Very unhealthy"


    elif (row.Region == 2):
        if (row.AQI <= 9):
            level = "Good"
            l_code = 0
        elif (row.AQI <= 14):
            level = "Moderate"
            l_code = 1
        elif (row.AQI<=18):
            level = "Unhealthy"
            l_code = 2
        else:
            level="Very unhealthy"
            l_code=3

    elif (row.Region==3):
        if (row.AQI<= 20):
            level = "Good"
            l_code = 0
        elif (row.AQI<= 28):
            level = "Moderate"
            l_code = 1
        elif (row.AQI <= 32):
            level = "Unhealthy"
            l_code = 2
        else:
            level="Very unhealthy"
            l_code=3

    level_of_concern_new.append(l_code)

print(level_of_concern_new)
new_dataframe.insert(7, "HealthConcern", level_of_concern_new, True)
print(new_dataframe)
#split dataset in features and target variable
feature_cols = ['Region','Time_GMT','Wday','O3','PM25','NO2','AQI']
X = new_dataframe[feature_cols] # Features
y = new_dataframe['HealthConcern'] # Target variable


train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.25, random_state = 14)
model = RandomForestRegressor(n_estimators = 500, random_state = 42)
model.fit(train_X,train_y)

# Get the mean absolute error on the validation data
predicted_AQHI_level= model.predict(val_X)
MAE = mean_absolute_error(val_y , predicted_AQHI_level)
print('\n\nRandom forest validation MAE = ', MAE)
print("\n\nRandom Forest - predicted_AQHI_level",predicted_AQHI_level)
# Calculate the absolute errors
errors = abs(predicted_AQHI_level- val_y)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = 100 * (errors/ val_y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
predicted_AQHI_level_test=model.predict(test_data_2)
print("\n\n Predicted AQHI level for testing value  using Random forest \n",predicted_AQHI_level)
make_submission(predicted_AQHI_level_test,'Submission(RF)_for AQI LevelPrediction.csv')

filename= 'train_AQHI_Level_model_by_randomForest.pkl'
joblib.dump(model, filename)

print("\n Decision Tree\n")
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.25)
DecisionTreeRegressor_model = DecisionTreeClassifier(max_leaf_nodes=1000, random_state=42,)
DecisionTreeRegressor_model.fit(train_X, train_y)
preds_level = DecisionTreeRegressor_model.predict(val_X)
mae = mean_absolute_error(val_y, preds_level)
print('\n\nDecision Tree validation MAE = ', mae)
print("\nDecision Tree - predicted_AQHI_level",preds_level)
# test_decision_tree=pd.read_csv('Test_data.csv')


predicted_AQHI_level_test_DT=DecisionTreeRegressor_model.predict(test_data_2)
print("\n\n Predicted AQHI level for testing value using  Descision tree \n",predicted_AQHI_level_test_DT)
make_submission(predicted_AQHI_level_test_DT,'Submission(Decision_Tree)_for AQI LevelPrediction.csv')


filename= 'train_AQHI_Level_model_by_DecisionTree.pkl'
joblib.dump(DecisionTreeRegressor_model, filename)


print("\n Evaluating the output \n")
for row in test_data_2.itertuples():

    if (row.Region == 1):
        if(row.Time_GMT>=8) &(row.Time_GMT <= 14) & (row.Wday==1):
            if(row.AQI<=10):
                level=='Good'
                l_code=0
            elif(row.AQI<=15):
                l_code=1
            elif(row.AQI<=19):
                l_code=2
            else:
                l_code=3
        else:
            if(row.AQI<=16):
                l_code=0
            elif(row.AQI<=21):
                l_code=1
            elif(row.AQI<=25):
                l_code=2
            else:
                l_code=3

    elif (row.Region == 2):

        if (row.AQI <= 9):

            level = "Good"

            l_code = 0

        elif (row.AQI <= 14):

            level = "Moderate"

            l_code = 1

        elif (row.AQI <= 18):

            level = "Unhealthy"

            l_code = 2

        else:

            level = "Very unhealthy"

            l_code = 3


    elif (row.Region == 3):

        if (row.AQI <= 20):

            level = "Good"

            l_code = 0

        elif (row.AQI <= 28):

            level = "Moderate"

            l_code = 1

        elif (row.AQI <= 32):

            level = "Unhealthy"

            l_code = 2

        else:

            level = "Very unhealthy"

            l_code = 3
    level_of_concern_actual.append(l_code)
print("\n level_of_concern_actual", level_of_concern_actual)