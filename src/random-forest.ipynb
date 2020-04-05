{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 13,
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
    "from sklearn.model_selection import RandomizedSearchCV\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reading data from a combined file with both good and failed drives\n",
    "def split_train_val_test_data( root = \"./\", drive_file = \"/ST12000NM0007_last_10_day.csv\",  \n",
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
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def random_tune_randomforest():\n",
    "    rf = RandomForestClassifier(random_state = 1)\n",
    "    # Number of trees in random forest\n",
    "    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]\n",
    "    # Number of features to consider at every split\n",
    "    max_features = ['auto', 'sqrt']\n",
    "    # Maximum number of levels in tree\n",
    "    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]\n",
    "    max_depth.append(None)\n",
    "    # Minimum number of samples required to split a node\n",
    "    min_samples_split = [2, 5, 10]\n",
    "    # Minimum number of samples required at each leaf node\n",
    "    min_samples_leaf = [1, 2, 4]\n",
    "    # Method of selecting samples for training each tree\n",
    "    bootstrap = [True, False]\n",
    "    #Entropy calculations\n",
    "    criterion = [\"gini\", \"entropy\"]\n",
    "    \n",
    "    random_grid = {'n_estimators': n_estimators,\n",
    "               'max_features': max_features,\n",
    "               'max_depth': max_depth,\n",
    "               'min_samples_split': min_samples_split,\n",
    "               'min_samples_leaf': min_samples_leaf,\n",
    "               'bootstrap': bootstrap,\n",
    "               'criterion': criterion\n",
    "                }\n",
    "    \n",
    "    rf_random = RandomizedSearchCV(\n",
    "        estimator = rf, \n",
    "        param_distributions = random_grid, \n",
    "        n_iter = 100, \n",
    "        cv = 3, \n",
    "        verbose=2, \n",
    "        random_state=1, \n",
    "        n_jobs = -1, \n",
    "        scoring = [\"f1\", \"accuracy\"], \n",
    "        refit=\"f1\"\n",
    "    )\n",
    "    \n",
    "    return rf_random\n",
    "    \n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run(models = [RandomForestClassifier(max_depth=2, random_state=0)], tune_model=False):\n",
    "    #X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_data(drive_file = \"/ST12000NM0007_last_day_all_q_raw.csv\", smote_data=True)\n",
    "    X_train, X_test, y_train, y_test = split_train_val_test_data(drive_file = \"/ST12000NM0007_last_day_all_q_raw.csv\", resample_data=True)\n",
    "    #X_train, X_test, y_train, y_test = get_train_test_data(resample_data=True)\n",
    "    print(\"Data loaded successfully...\\n\")\n",
    "    for model in models:  \n",
    "        print(\"\\n\\n *\", type(model).__name__)  \n",
    "        \n",
    "        if(type(model).__name__ == \"XGBClassifier\" and tune_model):\n",
    "            tune_xgb(model, X_train, y_train)\n",
    "\n",
    "        start = time()\n",
    "        model.fit(X_train, y_train)\n",
    "        end = time()\n",
    "        print(\"\\nTime to train:\", str((end - start)/60), \" mins\")\n",
    "        \n",
    "        print(model.best_params_)\n",
    "        # Test set results\n",
    "        print(\"\\n- Results on test set: \")\n",
    "        y_pred = model.predict(X_test)\n",
    "        print(\"Accuracy: \", accuracy_score(y_test, y_pred))\n",
    "        print(\"Scores:\\n\", classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data loaded successfully...\n",
      "\n",
      "\n",
      "\n",
      " * RandomizedSearchCV\n",
      "Fitting 3 folds for each of 100 candidates, totalling 300 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.\n",
      "[Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed: 18.3min\n",
      "[Parallel(n_jobs=-1)]: Done 154 tasks      | elapsed: 94.1min\n",
      "[Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 154.2min finished\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Time to train: 157.08657914797465  mins\n",
      "{'n_estimators': 2000, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 40, 'criterion': 'entropy', 'bootstrap': True}\n",
      "\n",
      "- Results on test set: \n",
      "Accuracy:  0.9849712493465761\n",
      "Scores:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.98      0.99      7425\n",
      "           1       0.66      1.00      0.80       227\n",
      "\n",
      "    accuracy                           0.98      7652\n",
      "   macro avg       0.83      0.99      0.90      7652\n",
      "weighted avg       0.99      0.98      0.99      7652\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Entry point of function\n",
    "if __name__ == \"__main__\":\n",
    "    models_list = []\n",
    "    rf = random_tune_randomforest()\n",
    "    models_list.append(rf)\n",
    "#     rfc = RandomForestClassifier(max_depth=2, random_state=0)\n",
    "#     models_list.append(rfc)\n",
    "    run(models_list,tune_model=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_rf_10(file_path=\"/ST12000NM0007_last_day_all_q_raw.csv\"):\n",
    "\n",
    "    model = RandomForestClassifier(\n",
    "        n_estimators = 2000, \n",
    "        min_samples_split = 5, \n",
    "        min_samples_leaf = 4,\n",
    "        max_features = 'auto', \n",
    "        max_depth = 40, \n",
    "        criterion = 'entropy',\n",
    "        bootstrap = True\n",
    "    )\n",
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
    "    print(\"Accuracy: \", accuracy_score(y_test, y_pred))\n",
    "    print(\"Scores:\\n\", classification_report(y_test, y_pred))\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data loaded successfully...\n",
      "\n",
      "\n",
      "\n",
      " * RandomForestClassifier\n",
      "\n",
      "Time to train: 19.21467758019765  mins\n",
      "\n",
      "- Results on test set: \n",
      "Accuracy:  0.9999357876712329\n",
      "Scores:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     45926\n",
      "           1       1.00      1.00      1.00       794\n",
      "\n",
      "    accuracy                           1.00     46720\n",
      "   macro avg       1.00      1.00      1.00     46720\n",
      "weighted avg       1.00      1.00      1.00     46720\n",
      "\n"
     ]
    }
   ],
   "source": [
    "run_rf_10(\"/ST4000DM000_last_10_day_all_q_raw.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data loaded successfully...\n",
      "\n",
      "\n",
      "\n",
      " * RandomForestClassifier\n",
      "\n",
      "Time to train: 5.710624718666077  mins\n",
      "\n",
      "- Results on test set: \n",
      "Accuracy:  1.0\n",
      "Scores:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     19627\n",
      "           1       1.00      1.00      1.00       239\n",
      "\n",
      "    accuracy                           1.00     19866\n",
      "   macro avg       1.00      1.00      1.00     19866\n",
      "weighted avg       1.00      1.00      1.00     19866\n",
      "\n"
     ]
    }
   ],
   "source": [
    "run_rf_10(\"/ST8000DM002_last_10_day_all_q_raw.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data loaded successfully...\n",
      "\n",
      "\n",
      "\n",
      " * RandomForestClassifier\n",
      "\n",
      "Time to train: 9.502617053190868  mins\n",
      "\n",
      "- Results on test set: \n",
      "Accuracy:  1.0\n",
      "Scores:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     28906\n",
      "           1       1.00      1.00      1.00       436\n",
      "\n",
      "    accuracy                           1.00     29342\n",
      "   macro avg       1.00      1.00      1.00     29342\n",
      "weighted avg       1.00      1.00      1.00     29342\n",
      "\n"
     ]
    }
   ],
   "source": [
    "run_rf_10(\"/ST8000NM0055_last_10_day_all_q_raw.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data loaded successfully...\n",
      "\n",
      "\n",
      "\n",
      " * RandomForestClassifier\n",
      "\n",
      "Time to train: 57.33980790376663  mins\n",
      "\n",
      "- Results on test set: \n",
      "Accuracy:  0.9997122753786194\n",
      "Scores:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     74210\n",
      "           1       0.99      1.00      1.00      2252\n",
      "\n",
      "    accuracy                           1.00     76462\n",
      "   macro avg       1.00      1.00      1.00     76462\n",
      "weighted avg       1.00      1.00      1.00     76462\n",
      "\n"
     ]
    }
   ],
   "source": [
    "run_rf_10(\"/ST12000NM0007_last_10_day_all_q_raw.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data loaded successfully...\n",
      "\n",
      "\n",
      "\n",
      " * RandomForestClassifier\n",
      "\n",
      "Time to train: 0.16820285320281983  mins\n",
      "\n",
      "- Results on test set: \n",
      "Accuracy:  0.9990636704119851\n",
      "Scores:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00       912\n",
      "           1       0.99      1.00      1.00       156\n",
      "\n",
      "    accuracy                           1.00      1068\n",
      "   macro avg       1.00      1.00      1.00      1068\n",
      "weighted avg       1.00      1.00      1.00      1068\n",
      "\n"
     ]
    }
   ],
   "source": [
    "run_rf_10(\"/TOSHIBA MQ01ABF050_last_10_day_all_q_raw.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Default parameters\n",
    "\n",
    "* XGBClassifier\n",
    "\n",
    "Time to train: 0.03838756481806437  mins\n",
    "\n",
    "- Results on validation set: \n",
    "Accuracy:  0.9845791949817041\n",
    "Scores:\n",
    "               precision    recall  f1-score   support\n",
    "\n",
    "           0       0.99      1.00      0.99      7425\n",
    "           1       0.85      0.59      0.69       227\n",
    "\n",
    "    accuracy                           0.98      7652\n",
    "   macro avg       0.92      0.79      0.84      7652\n",
    "weighted avg       0.98      0.98      0.98      7652\n",
    "\n",
    "\n",
    "- Results on test set: \n",
    "Accuracy:  0.9801359121798223\n",
    "Scores:\n",
    "               precision    recall  f1-score   support\n",
    "\n",
    "           0       0.98      1.00      0.99      7425\n",
    "           1       0.91      0.37      0.52       227\n",
    "\n",
    "    accuracy                           0.98      7652\n",
    "   macro avg       0.95      0.68      0.76      7652\n",
    "weighted avg       0.98      0.98      0.98      7652\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
   "version": "3.7.4"
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
