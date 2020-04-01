import pandas as pd
import numpy as np
from time import time

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from collections import Counter

#Sorting data
def sort_data_by_date(file_path):
    df = pd.read_csv(file_path, parse_dates=True)
    sorted_df = df.sort_values(["date"])
    return sorted_df

def split_train_test_data( root = "./dataset", drive_file = "/ST12000NM0007_last_10_day.csv",  ignore_cols = ["date","serial_number","model","capacity_bytes","failure"], resample_data=False, smote_data=False):

    df = pd.read_csv(root+drive_file, parse_dates=True)

    df_good = df.loc[df['failure'] == 0]
    df_bad = df.loc[df['failure'] == 1]
     
    df_good = df_good.sort_values(["date"])
    df_bad = df_bad.sort_values(["date"])

    good_y = df_good["failure"]
    bad_y = df_bad["failure"]

    # Take the first 70% of data as train and rest 30% as test
    X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(
        df_good, good_y, train_size=0.7, shuffle=False)
    X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(
        df_bad, bad_y, train_size=0.7, shuffle=False)
    print("Bad Y test count:", len(y_test_bad))
    print("Good Y test count:", len(y_test_good))


    if resample_data:
        X_train_bad = resample(df_bad, replace=True, n_samples=len(X_train_good), random_state=1)
        X_train_bad = X_train_bad.sort_values(["date"])

    y_train_bad = X_train_bad["failure"]

    X_train = pd.concat([X_train_good, X_train_bad], axis=0)
    X_test = pd.concat([X_test_good, X_test_bad], axis=0)
    y_train = pd.concat([y_train_good, y_train_bad], axis=0)
    y_test = pd.concat([y_test_good, y_test_bad], axis=0)

    X_train.drop(columns=ignore_cols, inplace=True, axis=1)
    X_test.drop(columns=ignore_cols, inplace=True, axis=1)

    if smote_data:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("LABEL COUNT: ", Counter(y_train))

    return (X_train, X_test, y_train, y_test)


def get_train_test_data(ignore_cols=["date", "serial_number", "model", "capacity_bytes", "failure"], resample_data=False, smote_data=False):
    data_root_dir = "./dataset"
    good_drives_file = "/k_only_good.csv"
    failed_drives_file = "/k_only_failed.csv"
    
    # Sort df by date
    good_drives = sort_data_by_date(data_root_dir+good_drives_file)
    failed_drives = sort_data_by_date(data_root_dir+failed_drives_file)

    print("Done reading data")
    good_y = good_drives["failure"]
    failed_y = failed_drives["failure"] 

    # Take the first 70% of data as train and rest 30% as test
    X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(good_drives, good_y, train_size = 0.7, shuffle=False)
    X_train_failed, X_test_failed, y_train_failed, y_test_failed = train_test_split(failed_drives, failed_y, train_size = 0.7, shuffle=False)
    print("Bad Y test count:", len(y_test_failed))
    print("Good Y test count:", len(y_test_good))

    #df.head(int(len(df)*(n/100)))
    if resample_data:
        X_train_failed = resample(X_train_failed, replace=True, n_samples=len(X_train_good), random_state=1)
        X_train_failed = X_train_failed.sort_values(["date"])
        print("Shape train for good drives: ", X_train_good.shape)
        print("Shape train for failed drives: ", X_train_failed.shape)

    y_train_failed = X_train_failed["failure"]

    # Concatenating the good and failed dataset to get final train and test dataset
    X_train = pd.concat([X_train_good, X_train_failed], axis = 0)
    X_test = pd.concat([X_test_good, X_test_failed], axis = 0)
    y_train = pd.concat([y_train_good, y_train_failed], axis = 0)
    y_test = pd.concat([y_test_good, y_test_failed], axis = 0)

    print("X train shape: ", X_train.shape)
    X_train.drop(columns = ignore_cols, inplace=True, axis=1)
    X_test.drop(columns = ignore_cols, inplace=True, axis=1)

    if smote_data:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    return (X_train, X_test, y_train, y_test)


def run(models = [RandomForestClassifier(max_depth=2, random_state=0)]):
    X_train, X_test, y_train, y_test = split_train_test_data(drive_file = "/ST12000NM0007_last_10_day.csv", smote_data=True)
    #X_train, X_test, y_train, y_test = get_train_test_data(resample_data=True)
    print("Got data!!")
    for model in models:  
        print(type(model).__name__)  
        start = time()
        model.fit(X_train, y_train)
        end = time()
        print("Time to train:", str((end - start)/60), " mins")
        y_pred = model.predict(X_test)

        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("Scores:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    models_list = []
    xgbc = XGBClassifier()
    models_list.append(xgbc)
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    models_list.append(rfc)
    # mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # models_list.append(mlpc)
    run(models_list)

# conda install -c conda-forge xgboost
# pip install xgboost
