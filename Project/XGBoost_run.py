import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import keras
from sklearn.model_selection import train_test_split
from preprocessing import preprocessingClass as preprocess
from regression_models import regressionModels
import xgboost as xg
df = pd.read_csv("historical_data.csv")
test_df=pd.read_csv("predict_data.csv")

df=df.iloc[:,1:]
preprocess=preprocess(df,test_df)

preprocess.handleNegativeValues(columns=['total_onshift_dashers','total_busy_dashers','total_outstanding_orders'],
                               substitution_value=0)

imputation_dict={'estimated_store_to_consumer_driving_duration':'mean','total_onshift_dashers':'median',
                'total_busy_dashers':'median','total_outstanding_orders':'median','market_id':'mode'}
preprocess.imputeMissingValues(imputation_dict=imputation_dict)

preprocess.dropRecordsWithNullValues()

preprocess.removeTargetOutliers('total_delivery_duration_seconds')

features_to_be_generated=['hour','month','day']
preprocess.generateFeaturesFromDateColumn('created_at',features_to_be_generated)

## Applying one hot encoding for Market_id column
preprocess.applyOneHotEncoding(['market_id'])

train_columns_to_be_dropped=['created_at','created_at_month','created_at_day','created_at_hour','market_id','actual_delivery_time']
test_columns_to_be_dropped=['created_at','created_at_month','created_at_day','created_at_hour','market_id','delivery_id']
preprocess.dropColumns(train_columns_to_be_dropped,test_columns_to_be_dropped)

df=preprocess.getTrainDataframe()
test_df=preprocess.getTestDataframe()

X=df.drop('total_delivery_duration_seconds',axis=1)
Y=df['total_delivery_duration_seconds']

X_train, X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)
X_test=copy.deepcopy(test_df)

X_train,X_val,X_test=preprocess.applyTargetEncoding(X_train,Y_train,X_val,X_test,columns=['store_id'])

columns_to_be_scaled = ['estimated_store_to_consumer_driving_duration',  'subtotal','total_onshift_dashers'
                   ,'total_busy_dashers','total_outstanding_orders','store_id']

X_train,X_val,X_test=preprocess.featureScaling(columns_to_be_scaled,X_train,X_val,X_test,option='standarization')

xgb_r = xg.XGBRegressor(objective='reg:linear',
                        n_estimators=100, max_depth=7, seed=123)

xgb_r.fit(X_train, Y_train)

Y_pred = xgb_r.predict(X_val)
Y_pred_test=xgb_r.predict(X_test)
test_df=pd.read_csv("predict_data.csv") # we will read the test file again as we dropped few columns from it during preprocessing
test_df['Predictions']=Y_pred_test
test_df.to_csv("prediction_file_XGBOOST.csv")
