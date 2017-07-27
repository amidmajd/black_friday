import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn import neighbors
sns.set()


data = pd.read_csv('train_cleaned.csv')
test = pd.read_csv('test_cleaned.csv')
target = data.Purchase
data = data.drop('Purchase', axis=1)


# tmp = []
# for i in range(10, 50):
#     print(i)
#     model = RandomForestRegressor(n_estimators=30, n_jobs=-1, max_depth=i)
#     model.fit(x_train, y_train)
#     tmp.append(model.score(x_test, y_test) * 100)
# plt.plot(range(10, 50), tmp)
# plt.show()

# cols = ['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
#        'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
#        'Product_Category_2', 'Product_Category_3', 'pid_4_f_lett_0',
#        'pid_4_f_lett_1', 'pid_4_f_lett_2', 'pid_4_f_lett_3', 'pid_4_f_lett_4',
#        'pid_2_l_lett', 'pid']

# for col in cols:
#     fig = plt.figure()
#     sns.regplot(data.loc[::100, col], target.iloc[::100])
# plt.show()

# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# tmp = []
# for i in range(5, 30):
#     model = RandomForestRegressor(n_estimators=30, n_jobs=-1, max_depth=i)
#     model.fit(x_train, y_train)
#     tmp.append(model.score(x_test, y_test) * 100)
#     print(i)
# plt.plot(range(5, 30), tmp)
# plt.show()
# print([(x+5, y) for x, y in enumerate(tmp)])

best_cols = ['City_Category', 'Product_Category_1', 'Product_Category_2',
             'Product_Category_3','pid_4_f_lett_1', 'pid']

model = RandomForestRegressor(n_jobs=-1, max_depth=20, n_estimators=30)
model.fit(data.loc[:, best_cols], target)

ft_imp = []
cols = best_cols
fig = plt.figure()
for i in range(len(cols)):
    ft_imp.append((cols[i], list(model.feature_importances_)[i]))
ft_imp = sorted(ft_imp, key=lambda x: x[1], reverse=True)
ft_imp = [(x, (y/sum([j for i, j in ft_imp])) * 100) for x, y in ft_imp]
sns.barplot([y for x, y in ft_imp], [x for x, y in ft_imp], orient='h')
fig.set_size_inches(16, 8)
# plt.show()
plt.savefig('feature_importance.png')


submission = pd.DataFrame(columns=['User_ID', 'Product_ID', 'Purchase'])
tmp_test = pd.read_csv('test.csv')
submission.User_ID = tmp_test.User_ID
submission.Product_ID = tmp_test.Product_ID

submission.Purchase = model.predict(test.loc[:, best_cols])
submission.to_csv('generated_sub.csv', index=False)
