import warnings
import numpy as np
import pandas as pd
import helper as hp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,  RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,  AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, validation_curve, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

hp.check_df(df)
df.drop('id', axis=1, inplace=True)
df['gender'].replace({'Other': 'Male'}, inplace=True)
cat_cols, num_cols, cat_but_car = hp.grab_col_names(df)

############        EDA         #########

######  KATEGORİK DEĞİŞKEN ANALİZİ  #########

for col in cat_cols:
    hp.cat_summary(df, col, 'stroke')


########    NÜMERİK DEĞİŞKENLERİN ANALİZİ       #########

for col in num_cols:
    hp.num_summary(df, col)


########    HEDEF DEĞİŞKEN ANALİZİ      ###########3

for col in cat_cols:
    hp.target_summary_with_cat(df, 'stroke', col)


#######     KORELASYON ANALİZİ          #########

hp.high_correlated_cols(df)


#######     EKSİK VE AYKIRI DEĞER ANALİZİ       #############
hp.missing_values_table(df)

#############   KNN en yakın komşu yaklaşımıyla eksik değerler dolduruldu

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df['bmi'] = dff['bmi']

num_cols = ['avg_glucose_level', 'bmi']

for col in num_cols:
    print(col,hp.check_outlier(df,col))

for col in num_cols:
    hp.replace_with_thresholds(df,col)


###############         FEATURE ENGINEERING         ################
df['work_type'].replace({'Never_worked': 'children'}, inplace=True)

df['avg_glucose_bmi'] = df['avg_glucose_level'] / df['bmi']
df['avg_glucose_plus_bmi'] = df['avg_glucose_level'] + df['bmi']

df['glucose_perc_age'] = df['avg_glucose_level'] / df['age']
df['bmi_perc_age'] = df['bmi'] / df['age']
df['danger_perc_age'] = df['avg_glucose_plus_bmi'] / df['age']

df.loc[(df['gender'] == 'Male'), 'gen_danger_perc_age'] = df['avg_glucose_plus_bmi'] / df['age']
df.loc[(df['gender'] == 'Female'), 'gen_danger_perc_age'] = df['avg_glucose_plus_bmi'] / df['age']
df.loc[(df['gender'] == 'Male'), 'gen_bmi_perc_age'] = df['bmi'] / df['age']
df.loc[(df['gender'] == 'Female'), 'gen_bmi_perc_age'] = df['bmi'] / df['age']

df['smoking_status'].replace({'Unknown': 0, 'never smoked': -1, 'formerly smoked': 1, 'smokes': 2 },
                             inplace=True)
df['danger_rate'] = df['hypertension'] + df['heart_disease'] + df['smoking_status']
df.groupby('danger_rate').agg({'stroke': ['mean','sum']})

df.drop('Residence_type', axis=1, inplace=True)

df2 = df.copy()
df = df2.copy()
df.head()

###############        SCALE VE ENCODING     ##############
cat_cols, num_cols, cat_but_car = hp.grab_col_names(df)

cat_cols = ['gender', 'work_type','ever_married']

df = hp.one_hot_encoder(df, cat_cols, drop_first=True)

for col in num_cols:
    print(col,hp.check_outlier(df,num_cols))

for col in num_cols:
    hp.replace_with_thresholds(df,col)

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

df.head()

###############        MODELING         ############
y = df['stroke']
X = df.drop('stroke', axis=1)

classifiers = [('LR', LogisticRegression()),
               ('KNN', KNeighborsClassifier()),
               ("SVC", SVC()),
               ("CART", DecisionTreeClassifier()),
               ("RF", RandomForestClassifier()),
               ('Adaboost', AdaBoostClassifier()),
               ('GBM', GradientBoostingClassifier())
               ]

for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X, y, cv=5, scoring=["roc_auc"])
    print(f"AUC: {round(cv_results['test_roc_auc'].mean(),7)} ({name}) ")


#############           HYPERMATER  TUNING          ####################3


gbm_model = GradientBoostingClassifier(random_state=17)
gbm_params = {"learning_rate": [0.01, 0.1],
             "max_depth": [3, 8],
             "n_estimators": [500, 1000],
             "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))


cv_results2 = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results2['test_accuracy'].mean()
cv_results2['test_f1'].mean()
cv_results2['test_roc_auc'].mean()

########################            FEATURE IMPORTANCES             ################


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(gbm_final,X)

gbm_final.score(X,y)