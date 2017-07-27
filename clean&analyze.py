import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from matplotlib import pyplot as plt
sns.set()


def main(mode):
    data = pd.read_csv(mode + '.csv')

    # cols = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
    #        'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
    #        'Product_Category_2', 'Product_Category_3', 'Purchase']


    # data.isnull().sum()

    # sns.regplot(data.loc[::100, 'User_ID'], data.Purchase.loc[::100])
    # plt.show()

    # sns.stripplot(data.loc[::100, 'Product_ID'], data.Purchase.loc[::100])
    # plt.show()

    # sns.stripplot(data.Gender.values, data.Purchase.values, jitter=True)
    # sns.stripplot(data.loc[::300, 'Gender'], data.Purchase.loc[::300], jitter=True)
    # plt.show()


    data.Gender = data.Gender.replace({'F': 0, 'M': 1})
    # sns.regplot(data.loc[::300, 'Gender'], data.Purchase.loc[::300])
    # plt.show()

    # sns.stripplot(data.loc[::200, 'Age'].values, data.Purchase.loc[::200'].values, jitter=True)
    # plt.show()

    # sns.regplot(data.loc[::200, 'Occupation'], data.Purchase.loc[::200])
    # plt.show()


    data.User_ID = (data.User_ID - data.User_ID.mean()) / data.User_ID.std()

    # print(list(data.Age.unique()))
    lb = preprocessing.LabelEncoder()
    lb.fit(['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])
    data.Age = lb.transform(data.Age)

    # print(list(data.City_Category.unique()))
    lb = preprocessing.LabelEncoder()
    lb.fit(['A', 'B', 'C'])
    data.City_Category = lb.transform(data.City_Category)

    # print(data.Stay_In_Current_City_Years.unique())
    lb = preprocessing.LabelEncoder().fit(['0', '1', '2', '3', '4+'])
    data.Stay_In_Current_City_Years = lb.transform(data.Stay_In_Current_City_Years)

    # sns.regplot(data.loc[::200, 'Product_Category_1'], data.Purchase.loc[::200])
    # plt.show()


    # len(data.Product_ID.unique()) # ==> 3631
    # pid = data.Product_ID.unique()
    # [x for x in pid if x[-2:] == '42']
    # ['P00375436', 'P00372445', 'P00370293', 'P00371644', 'P00370853']
    # data[data.Product_ID == 'P00085442'].Product_Category_2.isnull().unique()
    # this 5 Pids P_cat_2 & 3 are null

    # data.Product_ID
    # data.loc[[i for i,x in enumerate(data.Product_ID) if (x[:4]=='P000' and len(x)==8)]]
    # {'P000', 'P001', 'P002', 'P003', 'P009'}
    # data.loc[[1206,8]]


    data['pid_4_f_lett_0'] = np.zeros(len(data))
    data['pid_4_f_lett_1'] = np.zeros(len(data))
    data['pid_4_f_lett_2'] = np.zeros(len(data))
    data['pid_4_f_lett_3'] = np.zeros(len(data))
    data['pid_4_f_lett_4'] = np.zeros(len(data))
    data['pid_2_l_lett'] = np.zeros(len(data))
    data['pid'] = np.zeros(len(data))

    pid = data.Product_ID
    tmp_0 = []
    tmp_1 = []
    tmp_2 = []
    tmp_3 = []
    tmp_4 = []
    for i, x in enumerate(pid):
        if x[:4] == 'P000':
            tmp_0.append(i)
        elif x[:4] == 'P001':
            tmp_1.append(i)
        elif x[:4] == 'P002':
            tmp_2.append(i)
        elif x[:4] == 'P003':
            tmp_3.append(i)
        elif x[:4] == 'P009':
            tmp_4.append(i)
        else:
            print("UnExpected VALUE !!!!!!!!!!!!!!")

    data['pid_4_f_lett_0'].loc[tmp_0] = 1.0
    data['pid_4_f_lett_1'].loc[tmp_1] = 1.0
    data['pid_4_f_lett_2'].loc[tmp_2] = 1.0
    data['pid_4_f_lett_3'].loc[tmp_3] = 1.0
    data['pid_4_f_lett_4'].loc[tmp_4] = 1.0

    data.pid_2_l_lett = [x[-2:] for x in data.Product_ID]
    data.pid = [x[4:-2] for x in data.Product_ID]

    data = data.drop('Product_ID', axis=1)

    data.to_csv(mode + '_cleaned.csv', index=False)


if __name__ == '__main__':
    main('train')
    main('test')
