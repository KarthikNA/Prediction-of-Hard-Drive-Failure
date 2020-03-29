import pandas as pd
import numpy as np
from time import time

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#Sorting data
def sort_data_by_date(file_path):
    df = pd.read_csv(file_path, parse_dates=True)
    sorted_df = df.sort_values(["date"])
    return sorted_df

def get_train_test_data(ignore_cols = ["date","serial_number","model","capacity_bytes","failure"]):
    data_root_dir = "./dataset"
    good_drives_file = "/k_only_good.csv"
    failed_drives_file = "/k_only_failed.csv"
    
    # Sort df by date
    good_drives = sort_data_by_date(data_root_dir+good_drives_file)
    failed_drives = sort_data_by_date(data_root_dir+failed_drives_file)
    print("Done reading data")
    good_y = good_drives["failure"]
    failed_y = failed_drives["failure"] 

    good_drives.drop(columns = ignore_cols, inplace=True, axis=1)
    failed_drives.drop(columns = ignore_cols, inplace=True, axis=1)

    # Take the first 70% of data as train and rest 30% as test
    X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(good_drives, good_y, train_size = 0.7, shuffle=False)
    X_train_failed, X_test_failed, y_train_failed, y_test_failed = train_test_split(failed_drives, failed_y, train_size = 0.7, shuffle=False)

    #df.head(int(len(df)*(n/100)))

    # Concatenating the good and failed dataset to get final train and test dataset
    X_train = pd.concat([X_train_good, X_train_failed], axis = 0)
    X_test = pd.concat([X_test_good, X_test_failed], axis = 0)
    y_train = pd.concat([y_train_good, y_train_failed], axis = 0)
    y_test = pd.concat([y_test_good, y_test_failed], axis = 0)

    return (X_train, X_test, y_train, y_test)


def run(models = [RandomForestClassifier(max_depth=2, random_state=0)]):
    X_train, X_test, y_train, y_test = get_train_test_data()
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
    mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    models_list.append(mlpc)
    run(models_list)

# conda install -c conda-forge xgboost
# pip install xgboost
