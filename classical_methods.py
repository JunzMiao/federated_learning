# import pandas as pd
# # import seaborn as sns
# import numpy as np
# import sklearn
# import matplotlib.pyplot as plt
# import matplotlib as matplot
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split


# train = pd.read_csv("train-train.csv")
# # test = pd.read_csv("train-test.csv")

# x = train.drop(['attack', 'level', 'is_attack', 'attack_map'], axis=1)
# y = train.loc[:, ['is_attack']]


# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=908)


# from sklearn.ensemble import VotingClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import IsolationForest
# from sklearn.tree import DecisionTreeClassifier

# x = X_train
# y = y_train['is_attack'].ravel()

# clf1 = DecisionTreeClassifier() 
# clf2 = RandomForestClassifier(n_estimators=25, random_state=1)
# clf3 = GradientBoostingClassifier()
# ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False)

# eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('et',ET)], voting='hard') 

# for clf, label in zip([clf1, clf2, clf3, ET, eclf], ['DecisionTreeClassifier', 'Random Forest', 'GradientBoostingClassifier','ExtraTreesClassifier', 'Ensemble']): 
#     tmp = clf.fit(x,y)
#     pred = clf.score(X_test,y_test)
#     print("Acc: %0.2f [%s]" % (pred,label))
