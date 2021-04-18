import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

from fancyimpute import KNN
import time
import warnings
import gc

warnings.simplefilter(action='ignore', category=FutureWarning)
from xgboost import XGBClassifier


def classifier_selection(model, X_train_dataset, y_train_set, X_test_dataset):
    X_train_set=X_train_dataset.values
    X_test_set=X_test_dataset.values
    if model == 'xgboost':
        grid_xgboost = {
            'max_depth': range(2, 10, 1),
            'n_estimators': range(60, 220, 40),
            'learning_rate': [0.1, 0.01, 0.05]
        }
        grid_xgboost = GridSearchCV(XGBClassifier(), param_grid=grid_xgboost, cv=10, scoring='f1_macro', n_jobs=-1,
                                    refit=True)
        grid_xgboost.fit(X_train_set, y_train_set)
        model_xgboost = grid_xgboost.best_estimator_
        model_xgboost.fit(X_train_set, y_train_set)
        y_pred_train = model_xgboost.predict(X_train_set)
        y_pred_test = model_xgboost.predict(X_test_set)

    return y_pred_train, y_pred_test





timestr = time.strftime("%Y%m%d-%H%M%S")


data_set__list = ['australian']  # ,20.0,30.0,40.0,50.0,60.0,70.0,80.0]


results= []
results.append(['Dataset', 'experiment',
                "KNN_Accuracy_Train", "KNN_Matthews_Corrcoef_Train", "KNN_FScore_Micro_Train", "KNN_FScore_Macro_Train",
                "KNN_FScore_Weighted_Train", "KNN_Accucary_Test", "KNN_Matthews_Corrcoef_Test", "KNN_FScore_Micro_Test",
                "KNN_FScore_Macro_Test", "KNN_FScore_Weighted_Test",
                "MICE_Accuracy_Train", "MICE_Matthews_Validation_Train", "MICE_FScore_Micro_Train",
                "MICE_FScore_Macro_Train", "MICE_FScore_Weighted_Train", "MICE_Accuracy_Test",
                "MICE_Matthews_Validation_Test", "MICE_FScore_Micro_Test", "MICE_FScore_Macro_Test",
                "MICE_FScore_Weighted_Test",
                "SVD_Accuracy_Train", "SVD_Matthews_Validation_Train", "SVD_FScore_Micro_Train",
                "SVD_FScore_Macro_Train", "SVD_FScore_Weighted_Train", "SVD_Accuracy_Test",
                "SVD_Matthews_Validation_Test", "SVD_FScore_Micro_Test", "SVD_FScore_Macro_Test",
                "SVD_FScore_Weighted_Test",
                "Mean_Accuracy_Train", "Mean_Matthews_Validation_Train", "Mean_FScore_Micro_Train",
                "Mean_FScore_Macro_Train", "Mean_FScore_Weighted_Train", "Mean_Accuracy_Test",
                "Mean_Matthews_Validation_Test", "Mean_FScore_Micro_Test", "Mean_FScore_Macro_Test",
                "Mean_FScore_Weighted_Test",
                "Class_Mean_Accuracy_Train", "Class_Mean_Matthews_Validation_Train", "Class_Mean_FScore_Micro_Train",
                "Class_Mean_FScore_Macro_Train", "Class_Mean_FScore_Weighted_Train", "Class_Mean_Accuracy_Test",
                "Class_Mean_Matthews_Validation_Test", "Class_Mean_FScore_Micro_Test", "Class_Mean_FScore_Macro_Test",
                "Class_Mean_FScore_Weighted_Test",
                "XGboost_Accuracy_Train", "XGboost_Matthews_Corrcoef_Train", "XGboost_FScore_Micro_Train",
                "XGboost_FScore_Macro_Train", "XGboost_FScore_Weighted_Train", "XGboost_Accucary_Test",
                "XGboost_Matthews_Corrcoef_Test", "XGboost_FScore_Micro_Test", "XGboost_FScore_Macro_Test",
                "XGboost_FScore_Weighted_Test",
                ])


for dataset_name in data_set__list:
    for experiment in range(1, 2):
        X_train = pd.read_csv(dataset_name + '+MV-10-' + str(experiment) + "tra.dat", header=None)
        X_train.replace('<null>', np.nan, inplace=True)
        y_train = X_train.iloc[:, -1]  # last column as a target
        labels = y_train.unique().tolist()
        labels_transform = [i for i in range(0, len(labels))]
        y_train.replace(labels, labels_transform, inplace=True)
        X_train = X_train.iloc[:, 0:-1]
        X_test = pd.read_csv(dataset_name + '+MV-10-' + str(experiment) + "tst.dat", header=None)
        y_test = X_test.iloc[:, -1]
        y_test.replace(labels, labels_transform, inplace=True)
        X_test = X_test.iloc[:, 0:-1]
        le = preprocessing.LabelEncoder()
        if dataset_name=='german':
            column_list_nominal=[0,2,3,5,6,8,9,11,13,14,16,18,19]
            for i in column_list_nominal:
                labels_nominal=X_train[i].unique().tolist()
                labels_transform_nominal = [i for i in range(0, len(labels_nominal))]
                X_train[i].replace(labels_nominal, labels_transform_nominal, inplace=True)
                X_test[i].replace(labels_nominal, labels_transform_nominal, inplace=True)


        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_test = X_test.apply(pd.to_numeric, errors='coerce')



        # --------------------------xgboost--------------------------------------------#

        """y_pred_xgboost_train,y_pred_xgboost_test = classifier_selection('xgboost',X_train,y_train,X_test)

# --------------------------knn--------------------------------------------#

X_train_fill_KNN = KNN(k=1).fit_transform(X_train)
X_train_fill_KNN = pd.DataFrame(X_train_fill_KNN)
y_pred_knn_train, y_pred_knn_test = classifier_selection('xgboost', X_train_fill_KNN,y_train,X_test)


# --------------------------mice--------------------------------------------#

from fancyimpute import IterativeImputer


n_imputations = 5
X_train_fill_Iterative = []
for i in range(n_imputations):
    imputer = IterativeImputer(n_iter=5, sample_posterior=True, random_state=i)
    X_train_fill_Iterative.append(imputer.fit_transform(X_train))
X_train_fill_MICE = np.mean(X_train_fill_Iterative, 0)
X_train_fill_MICE = pd.DataFrame(X_train_fill_MICE)

y_pred_mice_train, y_pred_mice_test = classifier_selection('xgboost', X_train_fill_MICE,y_train, X_test)


# --------------------------svd--------------------------------------------#
from fancyimpute import SoftImpute, BiScaler


# X_incomplete_normalized = BiScaler().fit_transform(X_train)
X_train_fill_SVD = SoftImpute().fit_transform(X_train)
X_train_fill_SVD = pd.DataFrame(X_train_fill_SVD)
y_pred_SVD_train, y_pred_SVD_test = classifier_selection('xgboost', X_train_fill_SVD,y_train, X_test)"""

        # --------------------------mean--------------------------------------------#

        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_train_fill_mean = imp_mean.fit_transform(X_train)
        X_train_fill_mean = pd.DataFrame(X_train_fill_mean)
        y_pred_mean_train, y_pred_mean_test = classifier_selection('xgboost', X_train_fill_mean,y_train, X_test)

        # --------------------------classmean--------------------------------------------#

        X_train_fill_class_mean = X_train.copy()
        X = X_train.copy()
        X['target'] = y_train
        class_mean = [[X.iloc[:, j].groupby(X.iloc[:, -1]).mean().tolist()[i] for j in range(X_train.shape[1])] for
                      i in
                      (X['target'].unique().tolist())]
        print(class_mean)

        for i in range(X_train.shape[0]):
            for j in range(X_train.shape[1]):
                if pd.isnull(X_train.iloc[i, j]) == True:
                    class_index = y_train.iloc[i]
                    X_train_fill_class_mean.iloc[i, j] = class_mean[class_index][j]
                    if pd.isnull(X_train_fill_class_mean.iloc[i, j]) == True:
                        X_train_fill_class_mean.iloc[i, j] = X_train_fill_class_mean.iloc[:,j].mean()  # class mean degeri bulunumayan oznitelikler icin ortalama ile doldurma

        y_pred_class_mean_train, y_pred_class_mean_test = classifier_selection('xgboost', X_train_fill_class_mean, y_train, X_test)
        results.append([dataset_name, experiment,


                        accuracy_score(y_train, y_pred_mean_train),
                        matthews_corrcoef(y_train, y_pred_mean_train),
                        f1_score(y_train, y_pred_mean_train, average='micro'),
                        f1_score(y_train, y_pred_mean_train, average='macro'),
                        f1_score(y_train, y_pred_mean_train, average='weighted'),
                        accuracy_score(y_test, y_pred_mean_test),
                        matthews_corrcoef(y_test, y_pred_mean_test),
                        f1_score(y_test, y_pred_mean_test, average='micro'),
                        f1_score(y_test, y_pred_mean_test, average='macro'),
                        f1_score(y_test, y_pred_mean_test, average='weighted'),

                        accuracy_score(y_train, y_pred_class_mean_train),
                        matthews_corrcoef(y_train, y_pred_class_mean_train),
                        f1_score(y_train, y_pred_class_mean_train, average='micro'),
                        f1_score(y_train, y_pred_class_mean_train, average='macro'),
                        f1_score(y_train, y_pred_class_mean_train, average='weighted'),
                        accuracy_score(y_test, y_pred_class_mean_test),
                        matthews_corrcoef(y_test, y_pred_class_mean_test),
                        f1_score(y_test, y_pred_class_mean_test, average='micro'),
                        f1_score(y_test, y_pred_class_mean_test, average='macro'),
                        f1_score(y_test, y_pred_class_mean_test, average='weighted')

                        ])



result = pd.DataFrame(results)
result.to_excel('xgboost_comparedeneme'  + '.xlsx')




