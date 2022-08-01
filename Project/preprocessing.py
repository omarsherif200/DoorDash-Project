import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder

class preprocessingClass:
    def __init__(self,train_df,test_df):
        self.df=train_df
        self.test_df=test_df

    def setTrainDataframe(self ,train_df):
        self.df=train_df
    def setTestDataframe(self,test_df):
        self.test_df=test_df

    def getTrainDataframe(self):
        return self.df
    def getTestDataframe(self):
        return self.test_df

    def assertColumnsExistance(self,columns):
        for col in columns:
            if col not in self.df.columns:
                print(col +" doesn't exist in the dataframe")
                return False
        return True

    def handleNegativeValues(self , columns,substitution_value=0):
        if self.assertColumnsExistance(columns)==False:
            return
        for col in columns:
            self.df.loc[self.df[col] < 0, col] = substitution_value
            self.test_df.loc[self.test_df[col] < 0, col] = substitution_value

    def imputeMissingValues(self,imputation_dict):
        for col_name , imputation_option in imputation_dict.items():
            if col_name not in self.df.columns:
                print(col_name + " doesn't exist in the dataframe")
                continue
            if imputation_option.lower()=='median':
                median_val= self.df[col_name].median()
                self.df[col_name].fillna(median_val,inplace=True)
                self.test_df[col_name].fillna(median_val,inplace=True)
            elif imputation_option.lower()=='mean':
                mean_val = self.df[col_name].mean()
                self.df[col_name].fillna(mean_val, inplace=True)
                self.test_df[col_name].fillna(mean_val, inplace=True)
            elif imputation_option.lower()=='mode':
                mode_val = self.df[col_name].mode()[0]
                self.df[col_name].fillna(mode_val, inplace=True)
                self.test_df[col_name].fillna(mode_val, inplace=True)
            else:
                self.df[col_name].fillna(0, inplace=True)
                self.test_df[col_name].fillna(0, inplace=True)

    def dropDuplicateRecords(self):
        self.df.drop_duplicates(inplace=True)
        self.test_df.drop_duplicates(inplace=True)

    def dropRecordsWithNullValues(self):
        self.df.dropna(inplace=True)

    def removeTargetOutliers(self,target_col , tolerance=1500):
        if self.assertColumnsExistance([target_col]) == False:
            return
        Q3, Q1 = self.df[target_col].quantile(0.75), self.df[target_col].quantile(0.25)
        IQR = Q3 - Q1
        threshold=Q3+1.5*IQR + tolerance

        self.df = self.df.query('total_delivery_duration_seconds < @threshold')

    def applyOneHotEncoding(self,columns):
        if self.assertColumnsExistance(columns)==False:
            return

        oh_encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_market_col = pd.DataFrame(oh_encoder.fit_transform(self.df[columns]), index=self.df.index)
        encoded_market_col.columns = oh_encoder.get_feature_names()
        updated_df = pd.concat([self.df, encoded_market_col], axis=1)

        encoded_market_col_test = pd.DataFrame(oh_encoder.transform(self.test_df[columns]), index=self.test_df.index)
        encoded_market_col_test.columns = oh_encoder.get_feature_names()
        updated_test_df = pd.concat([self.test_df, encoded_market_col_test], axis=1)

        self.df,self.test_df=updated_df,updated_test_df

    def generateFeaturesFromDateColumn(self,col_name,features_to_be_generated):
        if self.assertColumnsExistance([col_name]) == False:
            return

        self.df[col_name] = pd.to_datetime(self.df[col_name])
        self.test_df[col_name] = pd.to_datetime(self.test_df[col_name])
        train_options={'month':self.df[col_name].dt.month,
                                  'day':self.df[col_name].dt.dayofweek,
                                  'hour':self.df[col_name].dt.hour}

        test_options=options={'month':self.test_df[col_name].dt.month,
                                  'day':self.test_df[col_name].dt.dayofweek,
                                  'hour':self.test_df[col_name].dt.hour}
        generated_features=[]
        for feature_name in features_to_be_generated:
            if feature_name.lower() in train_options.keys():
                generated_feature=col_name+'_'+feature_name.lower()
                self.df[generated_feature]=train_options[feature_name.lower()]
                self.test_df[generated_feature]=test_options[feature_name.lower()]
                generated_features.append(generated_feature)
            else:
                print(feature_name +" is not a valid option")

        self.applyOneHotEncoding(generated_features)

    def dropColumns(self,train_columns=None,test_columns=None):
        if train_columns !=None:
            if self.assertColumnsExistance(train_columns) == False:
                return
            self.df.drop(columns=train_columns, inplace=True)
        if test_columns != None:
            self.test_df.drop(columns=test_columns, inplace=True)

    def applyTargetEncoding(self,X_train,Y_train,X_val,X_test,columns):
        if self.assertColumnsExistance(columns) == False:
            return
        for col in columns:
            TargetEnc = TargetEncoder(cols=col)
            df_store = TargetEnc.fit_transform(X_train[col], Y_train)
            X_train_updated = df_store.join(X_train.drop(col, axis=1))

            df_store_val = TargetEnc.transform(X_val[col])
            X_val_updated = df_store_val.join(X_val.drop(col, axis=1))

            df_store_test = TargetEnc.transform(X_test[col])
            X_test_updated = df_store_test.join(X_test.drop(col, axis=1))
            return X_train_updated, X_val_updated, X_test_updated

    def featureScaling(self,columns,X_train,X_val,X_test,option='standarization'):
        if self.assertColumnsExistance(columns) == False:
            return
        scaler = None
        if option.lower()=='minmax':
            scaler=MinMaxScaler()
        else: ## standarization
            scaler = StandardScaler()
        X_train[columns] = scaler.fit_transform(X_train[columns])
        X_val[columns] = scaler.transform(X_val[columns])
        X_test[columns] = scaler.transform(X_test[columns])
        return X_train,X_val,X_test