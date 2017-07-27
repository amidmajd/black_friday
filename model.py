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


data = pd.read_csv('train_cleaned.csv')
test = pd.read_csv('test_cleaned.csv')
target = data.Purchase
data = data.drop('Purchase', axis=1)


# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# tmp = []
# for i in range(10, 50):
#     print(i)
#     model = RandomForestRegressor(n_estimators=30, n_jobs=-1, max_depth=i)
#     model.fit(x_train, y_train)
#     tmp.append(model.score(x_test, y_test) * 100)
# plt.plot(range(10, 50), tmp)
# plt.show()


# ft_imp = []
# cols = data.columns
# fig = plt.figure()
# for i in range(len(cols)):
#     ft_imp.append((cols[i], list(model.feature_importances_)[i]))
# ft_imp = sorted(ft_imp, key=lambda x: x[1], reverse=True)
# ft_imp = [(x, (y/sum([j for i, j in ft_imp])) * 100) for x, y in ft_imp]
# sns.barplot([y for x, y in ft_imp], [x for x, y in ft_imp], orient='h')
# fig.set_size_inches(16, 8)
# # plt.show()
# plt.savefig('feature_importance.png')

model = RandomForestRegressor(n_estimators=30, n_jobs=-1, max_depth=18)
model.fit(data, target)
# print(model.score(x_test, y_test))

submission = pd.DataFrame(columns=['User_ID', 'Product_ID', 'Purchase'])
tmp_test = pd.read_csv('test.csv')
submission.User_ID = tmp_test.User_ID
submission.Product_ID = tmp_test.Product_ID

submission.Purchase = model.predict(test)
submission.to_csv('generated_sub.csv', index=False)
