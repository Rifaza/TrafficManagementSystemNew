import pickle
from sklearn.externals import joblib
import  pandas as pd
from sklearn.preprocessing import  MinMaxScaler

mdl = joblib.load('/home/rifaza/Final_Year_Project/Spartans/Transportation/trainmodel.pkl')
test=pd.read_csv('/home/rifaza/Final_Year_Project/Spartans/Transportation/Final_Test.csv')
test=test.drop(['Id','Place_ID'],axis=1)
print(test.shape)
print(test)
sc=MinMaxScaler()
X_test=sc.fit_transform(test)
print(X_test)
y_pred=mdl.predict(test)
print(y_pred)
y_pred=pd.DataFrame(y_pred, columns=['AQI'])

# my_submission = pd.DataFrame({'Id': pd.read_csv('test.csv').Id, 'AQI': y_pred})
# my_submission.to_csv('{}.csv'.format('Final_outputfile'), index=False)
# output = pd.read_csv('Final_outputfile.csv')
test=pd.read_csv('test.csv')
level_of_concern = []


Region=test['Region']
print(type(Region))
for aqi in y_pred['AQI']:

    if (Region == 1).any():
        if ((test['Time_GMT'] >= 8).any() & (test['Time_GMT'] <= 14).any() & (test['Wday'] == 1).any()):

            if (aqi <= 3):
                level = "Good"
                l_code = 0
            elif (aqi <= 7):
                level = "Moderate"
                l_code = 1
            elif (aqi <=10):
                level = "Unhealthy"
                l_code = 2
            elif (aqi>10):
                level = " Very Unhealthy"
                l_code = 3

        else:
            if (aqi <= 3).any():
                level = "Good"
                l_code = 0
            elif (aqi <= 6).any():
                l_code = 1
                level = "Moderate"
            elif (aqi > 6).any():
                l_code = 2
                level = "Unhealthy"


    elif (Region == 2).any():
        if (aqi <= 2).any():
            level = "Good"
            l_code = 0
        elif (aqi <= 3).any():
            level = "Moderate"
            l_code = 1
        elif (aqi > 3).any():
            level = "Unhealthy"
            l_code = 2

    elif (Region==3).any() :
        if (aqi <= 5).any():
            level = "Good"
            l_code = 0
        elif (aqi <= 9).any():
            level = "Moderate"
            l_code = 1
        elif (aqi > 9).any():
            level = "Unhealthy"
            l_code = 2

    level_of_concern.append(level)

print(level_of_concern)
test.insert(8, "HealthConcern", level_of_concern, True)
print(test)

