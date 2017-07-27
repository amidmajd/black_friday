import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
sns.set()


def fill(mode):
    data = pd.read_csv(mode + '_cleaned.csv')

    # imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=1)
    # imputer.fit_transform(data)
    # data.Product_Category_2 = imputer.transform(data.Product_Category_2)[0]
    # data.Product_Category_3 = imputer.transform(data.Product_Category_3)[0]
    # 7176354
    # 71.5592233844

    # data.isnull().sum()
    # data.iloc[:,1:].head(20)

    not_nan = data[~data.Product_Category_2.isnull()].Product_Category_2
    nan = data[data.isnull()].Product_Category_2

    cols =['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
           'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
           'pid_4_f_lett_0',
           'pid_4_f_lett_1', 'pid_4_f_lett_2', 'pid_4_f_lett_3', 'pid_4_f_lett_4',
           'pid_2_l_lett', 'pid']
    # for col in cols:
    #     fig = plt.figure()
    #     sns.regplot(data[col].iloc[not_nan.index].iloc[::100], not_nan.iloc[::100])
    # plt.show()

    best_cols = ['Age',
                'Stay_In_Current_City_Years','Product_Category_1',
                'pid_4_f_lett_1', 'pid_4_f_lett_3', 'pid_4_f_lett_4',
                'pid']


    data_cat_2 = data.loc[:, best_cols][~ data.Product_Category_2.isnull()]
    data_cat_2_nan = data.loc[:, best_cols][data.Product_Category_2.isnull()]
    data_cat_3 = data.loc[:, best_cols][~data.Product_Category_3.isnull()]
    data_cat_3_nan = data.loc[:, best_cols][data.Product_Category_3.isnull()]

    target_cat_2 = data.loc[data_cat_2.index, 'Product_Category_2']
    target_cat_3 = data.loc[data_cat_3.index, 'Product_Category_3']


    # x_train, x_test, y_train, y_test = train_test_split(data_cat_2, target_cat_2, test_size=0.2)

    model_2 = RandomForestClassifier(max_depth=25, n_estimators=30, n_jobs=-1)
    model_2.fit(data_cat_2, target_cat_2)
    data.loc[data_cat_2_nan.index, 'Product_Category_2'] = model_2.predict(data_cat_2_nan)


    model_3 = RandomForestClassifier(max_depth=25, n_estimators=30, n_jobs=-1)
    model_3.fit(data_cat_3, target_cat_3)
    data.loc[data_cat_3_nan.index,'Product_Category_3'] = model_3.predict(data_cat_3_nan)

    data.to_csv(mode + '_cleaned.csv', index=False)


if __name__ == '__main__':
    fill('train')
    fill('test')
