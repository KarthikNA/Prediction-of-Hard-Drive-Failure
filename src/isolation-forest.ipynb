{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Packages\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from time import time\n",
    "\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "# from xgboost import XGBClassifier\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.utils import resample\n",
    "# from imblearn.over_sampling import SMOTE\n",
    "from collections import Counter\n",
    "from sklearn import metrics \n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.ensemble import IsolationForest\n",
    "\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import make_scorer\n",
    ""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reading data from a combined file with both good and failed drives\n",
    "def split_train_val_test_data( root = \"../dataset\", drive_file = \"/ST12000NM0007_last_10_day.csv\",  \n",
    "                          ignore_cols = [\"date\",\"serial_number\",\"model\",\"capacity_bytes\",\"failure\"], \n",
    "                          resample_data=False, smote_data=False):\n",
    "\n",
    "    df = pd.read_csv(root+drive_file, parse_dates=True)\n",
    "\n",
    "    df_good = df.loc[df['failure'] == 0]\n",
    "    df_bad = df.loc[df['failure'] == 1]\n",
    "     \n",
    "    df_good = df_good.sort_values([\"date\"])\n",
    "    df_bad = df_bad.sort_values([\"date\"])\n",
    "\n",
    "    good_y = df_good[\"failure\"]\n",
    "    bad_y = df_bad[\"failure\"]\n",
    "\n",
    "    # Split into train (80%) and test (20%)\n",
    "    X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(\n",
    "        df_good, good_y, train_size=0.8, shuffle=False)\n",
    "    X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(\n",
    "        df_bad, bad_y, train_size=0.8, shuffle=False)\n",
    "\n",
    "\n",
    "    # Split train into train and validation\n",
    "    # Train(60%), Val(20%), Test(20%)\n",
    "#     X_train_good, X_val_good, y_train_good, y_val_good = train_test_split(\n",
    "#         X_train_good, y_train_good, train_size=0.75, shuffle=False)\n",
    "#     X_train_bad, X_val_bad, y_train_bad, y_val_bad = train_test_split(\n",
    "#         X_train_bad, y_train_bad, train_size=0.75, shuffle=False)\n",
    "        \n",
    "    if resample_data:\n",
    "        X_train_bad = resample(df_bad, replace=True, n_samples=len(X_train_good), random_state=1)\n",
    "        X_train_bad = X_train_bad.sort_values([\"date\"])\n",
    "\n",
    "    y_train_bad = X_train_bad[\"failure\"]\n",
    "\n",
    "    X_train = pd.concat([X_train_good, X_train_bad], axis=0)\n",
    "    y_train = pd.concat([y_train_good, y_train_bad], axis=0)\n",
    "#     X_val = pd.concat([X_val_good, X_val_bad], axis=0)\n",
    "#     y_val = pd.concat([y_val_good, y_val_bad], axis=0)\n",
    "    X_test = pd.concat([X_test_good, X_test_bad], axis=0)\n",
    "    y_test = pd.concat([y_test_good, y_test_bad], axis=0)\n",
    "\n",
    "    X_train.drop(columns=ignore_cols, inplace=True, axis=1)\n",
    "#     X_val.drop(columns=ignore_cols, inplace=True, axis=1)\n",
    "    X_test.drop(columns=ignore_cols, inplace=True, axis=1)\n",
    "\n",
    "    if smote_data:\n",
    "        sm = SMOTE(random_state=42)\n",
    "        X_train, y_train = sm.fit_resample(X_train, y_train)\n",
    "\n",
    "    #return (X_train, X_val, X_test, y_train, y_val, y_test)\n",
    "    return (X_train, X_test, y_train, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sorting data by date\n",
    "def sort_data_by_date(file_path):\n",
    "    df = pd.read_csv(file_path, parse_dates=True)\n",
    "    sorted_df = df.sort_values([\"date\"])\n",
    "    return sorted_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def grid_tune_isolationforest():\n",
    "    iso = IsolationForest(random_state=0)\n",
    "    # Number of trees in random forest\n",
    "    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 5)]\n",
    "    # The number of samples to draw from X to train each base estimator.\n",
    "    max_samples = [0.2, 0.5, 0.8, 1]\n",
    "    # the proportion of outliers in the data set\n",
    "    contamination = ['auto', 0.1, 0.2, 0.3]\n",
    "    # The number of features to draw from X to train each base estimator.\n",
    "    max_features = [0.2, 0.5, 0.8, 1]\n",
    "    # Method of selecting samples for training each tree\n",
    "    bootstrap = [True, False]\n",
    "\n",
    "    \n",
    "    random_grid = {'n_estimators': n_estimators,\n",
    "               'max_features': max_features,\n",
    "               'bootstrap': bootstrap,\n",
    "               'max_samples': max_samples,\n",
    "               'contamination': contamination\n",
    "                }\n",
    "    def acc(y_true, y_pred): \n",
    "        y_pred = [0 if x > 0 else 1 for x in y_pred]\n",
    "        return accuracy_score(y_true, y_pred)\n",
    "    def f1(y_true, y_pred): \n",
    "        y_pred = [0 if x > 0 else 1 for x in y_pred]\n",
    "        return f1_score(y_true, y_pred)\n",
    "    iso_grid = GridSearchCV(\n",
    "        estimator = iso, \n",
    "        param_grid = random_grid, \n",
    "        cv = 3, \n",
    "        verbose=3, \n",
    "        n_jobs = -1, \n",
    "        scoring = {'f1': make_scorer(f1), 'acc': make_scorer(acc)}, \n",
    "        refit=\"f1\",\n",
    "        return_train_score=True\n",
    "    )\n",
    "    \n",
    "    return iso_grid\n",
    "    \n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run(models, tune_model=False):\n",
    "    #X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_data(drive_file = \"/ST12000NM0007_last_day_all_q_raw.csv\", smote_data=True)\n",
    "    X_train, X_test, y_train, y_test = split_train_val_test_data(drive_file = \"/ST12000NM0007_last_day_all_q_raw.csv\", resample_data=True)\n",
    "    print(\"Data loaded successfully...\\n\")\n",
    "    for model in models:  \n",
    "        print(\"\\n\\n *\", type(model).__name__)  \n",
    "        \n",
    "        # if(type(model).__name__ == \"XGBClassifier\" and tune_model):\n",
    "        #     tune_xgb(model, X_train, y_train)\n",
    "\n",
    "        start = time()\n",
    "        model.fit(X_train, y_train)\n",
    "        end = time()\n",
    "        print(\"\\nTime to train:\", str((end - start)/60), \" mins\")\n",
    "        \n",
    "        print(\"Best Parameter\", model.best_params_)\n",
    "        # Test set results\n",
    "        print(\"\\n- Results on test set: \")\n",
    "        y_pred = model.predict(X_test)\n",
    "        y_pred = y_pred = [0 if x > 0 else 1 for x in y_pred]\n",
    "        print(\"Accuracy: \", accuracy_score(y_test, y_pred))\n",
    "        print(\"Scores:\\n\", classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "Data loaded successfully...\n\n\n\n * GridSearchCV\nFitting 3 folds for each of 640 candidates, totalling 1920 fits\n[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.\n[Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:   20.2s\n[Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:  2.1min\n[Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:  5.0min\n[Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed: 10.9min\n[Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed: 18.5min\n[Parallel(n_jobs=-1)]: Done 1136 tasks      | elapsed: 27.0min\n[Parallel(n_jobs=-1)]: Done 1552 tasks      | elapsed: 37.8min\n[Parallel(n_jobs=-1)]: Done 1920 out of 1920 | elapsed: 48.0min finished\n\nTime to train: 48.03469951550166  mins\nBest Parameter {'bootstrap': True, 'contamination': 0.3, 'max_features': 1, 'max_samples': 0.8, 'n_estimators': 125}\n\n- Results on test set: \nAccuracy:  0.8691845269210664\nScores:\n               precision    recall  f1-score   support\n\n           0       0.98      0.88      0.93      7425\n           1       0.12      0.52      0.19       227\n\n    accuracy                           0.87      7652\n   macro avg       0.55      0.70      0.56      7652\nweighted avg       0.96      0.87      0.91      7652\n\n"
    }
   ],
   "source": [
    "# Entry point of function\n",
    "if __name__ == \"__main__\":\n",
    "    models_list = []\n",
    "    iso = grid_tune_isolationforest()\n",
    "    models_list.append(iso)\n",
    "    run(models_list,tune_model=True)\n",
    "'''\n",
    "Time to train: 48.03469951550166  mins\n",
    "Best Parameter {'bootstrap': True, 'contamination': 0.3, 'max_features': 1, 'max_samples': 0.8, 'n_estimators': 125}\n",
    "\n",
    "- Results on test set: \n",
    "Accuracy:  0.8691845269210664\n",
    "Scores:\n",
    "               precision    recall  f1-score   support\n",
    "\n",
    "           0       0.98      0.88      0.93      7425\n",
    "           1       0.12      0.52      0.19       227\n",
    "\n",
    "    accuracy                           0.87      7652\n",
    "   macro avg       0.55      0.70      0.56      7652\n",
    "weighted avg       0.96      0.87      0.91      7652\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_10(file_path=\"/ST12000NM0007_last_day_all_q_raw.csv\"):\n",
    "    model = IsolationForest(bootstrap= True, contamination= 0.3, max_features= 1, max_samples= 0.8, n_estimators= 125)\n",
    "    X_train, X_test, y_train, y_test = split_train_val_test_data(drive_file = file_path, resample_data=True)\n",
    "    \n",
    "    print(\"Data loaded successfully...\\n\")\n",
    "    print(\"\\n\\n *\", type(model).__name__)  \n",
    "\n",
    "    start = time()\n",
    "    model.fit(X_train, y_train)\n",
    "    end = time()\n",
    "    print(\"\\nTime to train:\", str((end - start)/60), \" mins\")\n",
    "    \n",
    "    # Test set results\n",
    "    print(\"\\n- Results on test set: \")\n",
    "    y_pred = model.predict(X_test)\n",
    "    y_pred = y_pred = [0 if x > 0 else 1 for x in y_pred]\n",
    "    print(\"Accuracy: \", accuracy_score(y_test, y_pred))\n",
    "    print(\"Scores:\\n\", classification_report(y_test, y_pred))\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "Data loaded successfully...\n\n\n\n * IsolationForest\n\nTime to train: 0.3185644030570984  mins\n\n- Results on test set: \nAccuracy:  0.6331121575342465\nScores:\n               precision    recall  f1-score   support\n\n           0       0.98      0.64      0.77     45926\n           1       0.01      0.13      0.01       794\n\n    accuracy                           0.63     46720\n   macro avg       0.49      0.39      0.39     46720\nweighted avg       0.96      0.63      0.76     46720\n\n"
    }
   ],
   "source": [
    "run_10(\"/ST4000DM000_last_10_day_all_q_raw.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "Data loaded successfully...\n\n\n\n * IsolationForest\n\nTime to train: 0.1331276337305705  mins\n\n- Results on test set: \nAccuracy:  0.40264773985704216\nScores:\n               precision    recall  f1-score   support\n\n           0       0.98      0.41      0.57     19627\n           1       0.00      0.20      0.01       239\n\n    accuracy                           0.40     19866\n   macro avg       0.49      0.30      0.29     19866\nweighted avg       0.96      0.40      0.57     19866\n\n"
    }
   ],
   "source": [
    "run_10(\"/ST8000DM002_last_10_day_all_q_raw.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "Data loaded successfully...\n\n\n\n * IsolationForest\n\nTime to train: 0.21579244534174602  mins\n\n- Results on test set: \nAccuracy:  0.5355463158612228\nScores:\n               precision    recall  f1-score   support\n\n           0       0.98      0.54      0.70     28906\n           1       0.01      0.24      0.01       436\n\n    accuracy                           0.54     29342\n   macro avg       0.49      0.39      0.36     29342\nweighted avg       0.96      0.54      0.69     29342\n\n"
    }
   ],
   "source": [
    "run_10(\"/ST8000NM0055_last_10_day_all_q_raw.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "Data loaded successfully...\n\n\n\n * IsolationForest\n\nTime to train: 0.7989759643872579  mins\n\n- Results on test set: \nAccuracy:  0.6798932803222516\nScores:\n               precision    recall  f1-score   support\n\n           0       0.98      0.69      0.81     74210\n           1       0.05      0.50      0.08      2252\n\n    accuracy                           0.68     76462\n   macro avg       0.51      0.59      0.45     76462\nweighted avg       0.95      0.68      0.78     76462\n\n"
    }
   ],
   "source": [
    "run_10(\"/ST12000NM0007_last_10_day_all_q_raw.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "Data loaded successfully...\n\n\n\n * IsolationForest\n\nTime to train: 0.006784852345784505  mins\n\n- Results on test set: \nAccuracy:  0.48408239700374533\nScores:\n               precision    recall  f1-score   support\n\n           0       0.82      0.51      0.63       912\n           1       0.11      0.35      0.17       156\n\n    accuracy                           0.48      1068\n   macro avg       0.46      0.43      0.40      1068\nweighted avg       0.72      0.48      0.56      1068\n\n"
    }
   ],
   "source": [
    "run_10(\"/TOSHIBA MQ01ABF050_last_10_day_all_q_raw.csv\")"
   ]
  }
 ],
 "metadata": {
  "file_extension": ".py",
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5-final"
  },
  "mimetype": "text/x-python",
  "name": "python",
  "npconvert_exporter": "python",
  "pygments_lexer": "ipython3",
  "version": 3
 },
 "nbformat": 4,
 "nbformat_minor": 2
}