{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Packages\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from time import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import metrics\n",
    "from sklearn.cluster import KMeans, DBSCAN, SpectralClustering\n",
    "from sklearn.decomposition import PCA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(38256, 20)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_csv(\"../dataset/ST12000NM0007_last_day_all_q_normalised.csv\")\n",
    "dataset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "drop_columns = [\"date\", \"serial_number\", \"model\", \"capacity_bytes\", \"failure\"]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset_copy = dataset.copy() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(38256, 15)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset_copy.drop(drop_columns, axis = 1, inplace=True)\n",
    "dataset_copy.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,\n",
       "       n_clusters=5, n_init=10, n_jobs=None, precompute_distances='auto',\n",
       "       random_state=None, tol=0.0001, verbose=0)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "kmeans = KMeans(n_clusters=5)\n",
    "kmeans.fit(dataset_copy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 80.98244525  94.08280876  99.95859562  84.44365635  89.33628868\n",
      "   99.78925437 100.          67.5915418   99.60980583  32.4084582\n",
      "   81.00629488  99.99663091  99.99663091 200.          99.98572568]\n",
      " [ 80.56148867  90.78543689  99.97152104  85.73009709  82.25760518\n",
      "   99.91197411 100.          69.03559871  99.49223301  30.96440129\n",
      "   35.09223301 100.         100.         200.         100.        ]\n",
      " [ 81.63807313  92.25264074  99.96401625  86.03755078  83.65455601\n",
      "   99.84428323 100.          75.18786999  99.87916425  24.81213001\n",
      "   81.64770749  99.99941962  99.99941962 200.         100.        ]\n",
      " [ 79.00990099  91.8019802   87.30693069  84.30693069  85.92079208\n",
      "   19.32673267 100.          71.36633663  99.93069307  28.63366337\n",
      "   78.28712871  99.2970297   99.2970297  200.         100.        ]\n",
      " [ 73.21979866  92.54408176  99.95530811  85.72574741  84.72528981\n",
      "   99.78172666 100.          72.87034777  99.79942038  27.12965223\n",
      "   73.25701647  99.99832215  99.99832215 200.         100.        ]]\n",
      "[2 2 4 ... 4 0 0]\n"
     ]
    }
   ],
   "source": [
    "centroids = kmeans.cluster_centers_\n",
    "labels = kmeans.labels_\n",
    "\n",
    "print(centroids)\n",
    "print(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4], dtype=int32)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.unique(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import matplotlib.pyplot as plt\n",
    "# from matplotlib import style\n",
    "# style.use(\"ggplot\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "38256"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# len(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# colors = [\"g.\",\"r.\",\"c.\",\"y.\"]\n",
    "# print(\"function started\")\n",
    "# for index, row in dataset.iterrows():\n",
    "# #     print(\"{} {}\".format(row[\"serial_number\"], labels[index]))\n",
    "#     plt.plot(row[\"serial_number\"], labels[index], colors[labels[index]], markersize = 10)\n",
    "# print(\"function stopped\")\n",
    "# plt.scatter(centroids[:, 0],centroids[:, 1], marker = \"x\", s=150, linewidths = 5, zorder = 10)\n",
    "# plt.show()    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# plt.scatter(centroids[:, 0],centroids[:, 1], marker = \"x\", s=150, linewidths = 5, zorder = 10)\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "# colors = [\"g.\",\"r.\",\"c.\",\"y.\"]\n",
    "\n",
    "# for index, row in dataset.iterrows()[:100]:\n",
    "# #     print(\"coordinate:\",dataset[i], \"label:\", labels[i])\n",
    "#     print(\"{} {}\".format(row[\"serial_number\"], labels[index]))\n",
    "#     plt.plot(row[\"serial_number\"], labels[index], colors[labels[i]], markersize = 10)\n",
    "\n",
    "\n",
    "# plt.scatter(centroids[:, 0],centroids[:, 1], marker = \"x\", s=150, linewidths = 5, zorder = 10)\n",
    "\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# centroids.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plt.scatter([i for i in range(len(labels))],labels, marker = \"x\", s=150, linewidths = 5, zorder = 10)\n",
    "# plt.show()"
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
      "6542\n",
      "31714\n"
     ]
    }
   ],
   "source": [
    "# li = []\n",
    "# l2 = []\n",
    "# for index, row in dataset.iterrows():\n",
    "#     if labels[index] == 1:\n",
    "# #         print(\"Serial Number : {} Cluster : {}\".format(row[\"serial_number\"], labels[index]))\n",
    "#         li.append(row[\"serial_number\"])\n",
    "#     else:\n",
    "#         l2.append(row[\"serial_number\"])\n",
    "# print(len(li))\n",
    "# print(len(l2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['date', 'serial_number', 'model', 'capacity_bytes', 'failure',\n",
       "       'smart_1_normalized', 'smart_3_normalized', 'smart_5_normalized',\n",
       "       'smart_7_normalized', 'smart_9_normalized', 'smart_187_normalized',\n",
       "       'smart_188_normalized', 'smart_190_normalized', 'smart_193_normalized',\n",
       "       'smart_194_normalized', 'smart_195_normalized', 'smart_197_normalized',\n",
       "       'smart_198_normalized', 'smart_199_normalized', 'smart_200_normalized'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True\n"
     ]
    }
   ],
   "source": [
    "if set(['smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']).issubset(dataset.columns):\n",
    "    print(True)\n",
    "else:\n",
    "    print(False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['date', 'serial_number', 'model', 'capacity_bytes', 'failure',\n",
       "       'smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw',\n",
       "       'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_188_raw',\n",
       "       'smart_190_raw', 'smart_192_raw', 'smart_193_raw', 'smart_194_raw',\n",
       "       'smart_195_raw', 'smart_197_raw', 'smart_198_raw', 'smart_199_raw',\n",
       "       'smart_240_raw', 'smart_241_raw', 'smart_242_raw'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_csv(\"../dataset/ST12000NM0007_last_day_all_q_raw.csv\")\n",
    "dataset.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "req_models = ['ST12000NM0007', 'ST4000DM000', 'ST8000NM0055', 'ST8000DM002', 'TOSHIBA MQ01ABF050', 'HGST HMS5C4040BLE640']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def has_req_cols(name):\n",
    "    full_name = name + \"_last_day_all_q_raw.csv\"\n",
    "    dataset = pd.read_csv(\"../dataset/\" + full_name)\n",
    "    if set(['smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']).issubset(dataset.columns):\n",
    "        print(\"{} has ALL Columns\".format(name))\n",
    "    else:\n",
    "        print(\"{} has does not have ALL Columns\".format(name))"
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
      "ST12000NM0007 has ALL Columns\n",
      "ST4000DM000 has ALL Columns\n",
      "ST8000NM0055 has ALL Columns\n",
      "ST8000DM002 has ALL Columns\n",
      "TOSHIBA MQ01ABF050 has does not have ALL Columns\n",
      "HGST HMS5C4040BLE640 has does not have ALL Columns\n"
     ]
    }
   ],
   "source": [
    "for i in req_models:\n",
    "    has_req_cols(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['date', 'serial_number', 'model', 'capacity_bytes', 'failure',\n",
       "       'smart_3_raw', 'smart_4_raw', 'smart_5_raw', 'smart_9_raw',\n",
       "       'smart_12_raw', 'smart_191_raw', 'smart_192_raw', 'smart_193_raw',\n",
       "       'smart_194_raw', 'smart_196_raw', 'smart_197_raw', 'smart_199_raw',\n",
       "       'smart_222_raw', 'smart_226_raw'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_csv(\"../dataset/TOSHIBA MQ01ABF050_last_day_all_q_raw.csv\")\n",
    "dataset.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['date', 'serial_number', 'model', 'capacity_bytes', 'failure',\n",
       "       'smart_1_raw', 'smart_2_raw', 'smart_3_raw', 'smart_4_raw',\n",
       "       'smart_5_raw', 'smart_7_raw', 'smart_8_raw', 'smart_9_raw',\n",
       "       'smart_10_raw', 'smart_12_raw', 'smart_192_raw', 'smart_193_raw',\n",
       "       'smart_194_raw', 'smart_196_raw', 'smart_197_raw', 'smart_199_raw'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_csv(\"../dataset/HGST HMS5C4040BLE640_last_day_all_q_raw.csv\")\n",
    "dataset.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "st_models = req_models = ['ST12000NM0007', 'ST4000DM000', 'ST8000NM0055', 'ST8000DM002']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_final = pd.DataFrame()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(86265, 8)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for name in st_models:\n",
    "    full_name = name + \"_last_day_all_q_raw.csv\"\n",
    "    dataset = pd.read_csv(\"../dataset/\" + full_name)\n",
    "    df = dataset[['model', 'serial_number','capacity_bytes', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']]\n",
    "    df_final = df_final.append(df, ignore_index=True)\n",
    "df_final.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(86265, 6)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_final_copy = df_final.copy()\n",
    "drop_columns = ['model', 'serial_number']\n",
    "df_final_copy.drop(drop_columns, axis = 1, inplace=True)\n",
    "df_final_copy.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,\n",
       "       n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',\n",
       "       random_state=None, tol=0.0001, verbose=0)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "kmeans = KMeans(n_clusters=4)\n",
    "kmeans.fit(df_final_copy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['capacity_bytes', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw',\n",
       "       'smart_197_raw', 'smart_198_raw'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_final_copy.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "centroids = kmeans.cluster_centers_\n",
    "labels = kmeans.labels_\n",
    "df_new = pd.DataFrame()\n",
    "df_new[\"model\"] = df_final[\"model\"]\n",
    "df_new[\"serial_number\"] = df_final[\"serial_number\"] \n",
    "df_new[\"labels\"] = labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "86265"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(86265, 3)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_new.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th>serial_number</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>model</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST12000NM0007</th>\n",
       "      <th>1</th>\n",
       "      <td>38207</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST4000DM000</th>\n",
       "      <th>0</th>\n",
       "      <td>23394</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000DM002</th>\n",
       "      <th>2</th>\n",
       "      <td>9936</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000NM0055</th>\n",
       "      <th>2</th>\n",
       "      <td>14679</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      serial_number\n",
       "model         labels               \n",
       "ST12000NM0007 1               38207\n",
       "              3                  49\n",
       "ST4000DM000   0               23394\n",
       "ST8000DM002   2                9936\n",
       "ST8000NM0055  2               14679"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_new.groupby(['model', 'labels']).count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th>serial_number</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>model</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST12000NM0007</th>\n",
       "      <th>1</th>\n",
       "      <td>38207</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST4000DM000</th>\n",
       "      <th>2</th>\n",
       "      <td>23394</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000DM002</th>\n",
       "      <th>0</th>\n",
       "      <td>9936</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000NM0055</th>\n",
       "      <th>0</th>\n",
       "      <td>14679</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      serial_number\n",
       "model         labels               \n",
       "ST12000NM0007 1               38207\n",
       "              3                  49\n",
       "ST4000DM000   2               23394\n",
       "ST8000DM002   0                9936\n",
       "ST8000NM0055  0               14679"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def kmeans():\n",
    "    st_models = req_models = ['ST12000NM0007', 'ST4000DM000', 'ST8000NM0055', 'ST8000DM002']\n",
    "    df_final = pd.DataFrame()\n",
    "    for name in st_models:\n",
    "        full_name = name + \"_last_day_all_q_raw.csv\"\n",
    "        dataset = pd.read_csv(\"../dataset/\" + full_name)\n",
    "        df = dataset[['model', 'serial_number','capacity_bytes', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']]\n",
    "        df_final = df_final.append(df, ignore_index=True)\n",
    "    df_final_copy = df_final.copy()\n",
    "    drop_columns = ['model', 'serial_number']\n",
    "    df_final_copy.drop(drop_columns, axis = 1, inplace=True)    \n",
    "    kmeans = KMeans(n_clusters=4)\n",
    "    kmeans.fit(df_final_copy)\n",
    "    centroids = kmeans.cluster_centers_\n",
    "    labels = kmeans.labels_\n",
    "    df_new = pd.DataFrame()\n",
    "    df_new[\"model\"] = df_final[\"model\"]\n",
    "    df_new[\"serial_number\"] = df_final[\"serial_number\"] \n",
    "    df_new[\"labels\"] = labels\n",
    "    dx = df_new.groupby(['model', 'labels']).count()\n",
    "    return dx\n",
    "result = kmeans()    \n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr:last-of-type th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th colspan=\"4\" halign=\"left\">serial_number</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>labels</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>model</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>ST12000NM0007</th>\n",
       "      <td>NaN</td>\n",
       "      <td>38207.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>49.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST4000DM000</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>23394.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000DM002</th>\n",
       "      <td>9936.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000NM0055</th>\n",
       "      <td>14679.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              serial_number                        \n",
       "labels                    0        1        2     3\n",
       "model                                              \n",
       "ST12000NM0007           NaN  38207.0      NaN  49.0\n",
       "ST4000DM000             NaN      NaN  23394.0   NaN\n",
       "ST8000DM002          9936.0      NaN      NaN   NaN\n",
       "ST8000NM0055        14679.0      NaN      NaN   NaN"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result.unstack()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dataset read\n",
      "dataset read complete\n",
      "dataset column drop complete\n",
      "algo started\n",
      "algo finished\n",
      "time return result\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th>serial_number</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>model</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST12000NM0007</th>\n",
       "      <th>-1</th>\n",
       "      <td>2952</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>35304</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST4000DM000</th>\n",
       "      <th>-1</th>\n",
       "      <td>1921</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>21473</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST8000DM002</th>\n",
       "      <th>-1</th>\n",
       "      <td>769</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>9167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST8000NM0055</th>\n",
       "      <th>-1</th>\n",
       "      <td>1220</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13459</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      serial_number\n",
       "model         labels               \n",
       "ST12000NM0007 -1               2952\n",
       "               0              35304\n",
       "ST4000DM000   -1               1921\n",
       "               1              21473\n",
       "ST8000DM002   -1                769\n",
       "               2               9167\n",
       "ST8000NM0055  -1               1220\n",
       "               2              13459"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def dbscan():\n",
    "    st_models = req_models = ['ST12000NM0007', 'ST4000DM000', 'ST8000NM0055', 'ST8000DM002']\n",
    "    df_final = pd.DataFrame()\n",
    "    print(\"dataset read\")\n",
    "    for name in st_models:\n",
    "        full_name = name + \"_last_day_all_q_raw.csv\"\n",
    "        dataset = pd.read_csv(\"../dataset/\" + full_name)\n",
    "        df = dataset[['model', 'serial_number','capacity_bytes', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']]\n",
    "        df_final = df_final.append(df, ignore_index=True)\n",
    "    print(\"dataset read complete\")    \n",
    "    df_final_copy = df_final.copy()\n",
    "    drop_columns = ['model', 'serial_number']\n",
    "    df_final_copy.drop(drop_columns, axis = 1, inplace=True)    \n",
    "    print(\"dataset column drop complete\")\n",
    "    print(\"algo started\")\n",
    "    db = DBSCAN( min_samples=1000)\n",
    "    db.fit(df_final_copy)\n",
    "    print(\"algo finished\")\n",
    "    labels = db.labels_\n",
    "    df_new = pd.DataFrame()\n",
    "    df_new[\"model\"] = df_final[\"model\"]\n",
    "    df_new[\"serial_number\"] = df_final[\"serial_number\"] \n",
    "    df_new[\"labels\"] = labels\n",
    "    dx = df_new.groupby(['model', 'labels']).count()\n",
    "    print(\"time return result\")\n",
    "    return dx\n",
    "result = dbscan()    \n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr:last-of-type th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th colspan=\"4\" halign=\"left\">serial_number</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>labels</th>\n",
       "      <th>-1</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>model</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>ST12000NM0007</th>\n",
       "      <td>2952.0</td>\n",
       "      <td>35304.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST4000DM000</th>\n",
       "      <td>1921.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>21473.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000DM002</th>\n",
       "      <td>769.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>9167.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000NM0055</th>\n",
       "      <td>1220.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>13459.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              serial_number                           \n",
       "labels                   -1        0        1        2\n",
       "model                                                 \n",
       "ST12000NM0007        2952.0  35304.0      NaN      NaN\n",
       "ST4000DM000          1921.0      NaN  21473.0      NaN\n",
       "ST8000DM002           769.0      NaN      NaN   9167.0\n",
       "ST8000NM0055         1220.0      NaN      NaN  13459.0"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result.unstack()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dataset read\n",
      "dataset read complete\n",
      "dataset column drop complete\n",
      "algo started\n"
     ]
    }
   ],
   "source": [
    "def spectral():\n",
    "    st_models = req_models = ['ST12000NM0007', 'ST4000DM000', 'ST8000NM0055', 'ST8000DM002']\n",
    "    df_final = pd.DataFrame()\n",
    "    print(\"dataset read\")\n",
    "    for name in st_models:\n",
    "        full_name = name + \"_last_day_all_q_raw.csv\"\n",
    "        dataset = pd.read_csv(\"../dataset/\" + full_name)\n",
    "        df = dataset[['model', 'serial_number','capacity_bytes', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']]\n",
    "        df_final = df_final.append(df, ignore_index=True)\n",
    "    print(\"dataset read complete\")    \n",
    "    df_final_copy = df_final.copy()\n",
    "    drop_columns = ['model', 'serial_number']\n",
    "    df_final_copy.drop(drop_columns, axis = 1, inplace=True)    \n",
    "    print(\"dataset column drop complete\")\n",
    "    print(\"algo started\")\n",
    "    sc = SpectralClustering(n_clusters=4)\n",
    "    sc.fit(df_final_copy)\n",
    "    print(\"algo finished\")\n",
    "    labels = sc.labels_\n",
    "    df_new = pd.DataFrame()\n",
    "    df_new[\"model\"] = df_final[\"model\"]\n",
    "    df_new[\"serial_number\"] = df_final[\"serial_number\"] \n",
    "    df_new[\"labels\"] = labels\n",
    "    dx = df_new.groupby(['model', 'labels']).count()\n",
    "    print(\"time return result\")\n",
    "    return dx\n",
    "result = spectral()    \n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "result.unstack()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "st_models = req_models = ['ST12000NM0007', 'ST4000DM000', 'ST8000NM0055', 'ST8000DM002']\n",
    "df_final = pd.DataFrame()\n",
    "for name in st_models:\n",
    "    full_name = name + \"_last_day_all_q_raw.csv\"\n",
    "    dataset = pd.read_csv(\"../dataset/\" + full_name)\n",
    "    df = dataset[['model', 'serial_number','capacity_bytes', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']]\n",
    "    df_final = df_final.append(df, ignore_index=True)\n",
    "df_final.to_csv(\"four_model_data.csv\")    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Automatically created module for IPython interactive environment\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 2100x1250 with 60 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(__doc__)\n",
    "\n",
    "import time\n",
    "import warnings\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn import cluster, datasets, mixture\n",
    "from sklearn.neighbors import kneighbors_graph\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from itertools import cycle, islice\n",
    "\n",
    "np.random.seed(0)\n",
    "\n",
    "# ============\n",
    "# Generate datasets. We choose the size big enough to see the scalability\n",
    "# of the algorithms, but not too big to avoid too long running times\n",
    "# ============\n",
    "n_samples = 1500\n",
    "noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,\n",
    "                                      noise=.05)\n",
    "noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)\n",
    "blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)\n",
    "no_structure = np.random.rand(n_samples, 2), None\n",
    "\n",
    "# Anisotropicly distributed data\n",
    "random_state = 170\n",
    "X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)\n",
    "transformation = [[0.6, -0.6], [-0.4, 0.8]]\n",
    "X_aniso = np.dot(X, transformation)\n",
    "aniso = (X_aniso, y)\n",
    "\n",
    "# blobs with varied variances\n",
    "varied = datasets.make_blobs(n_samples=n_samples,\n",
    "                             cluster_std=[1.0, 2.5, 0.5],\n",
    "                             random_state=random_state)\n",
    "\n",
    "# ============\n",
    "# Set up cluster parameters\n",
    "# ============\n",
    "plt.figure(figsize=(9 * 2 + 3, 12.5))\n",
    "plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,\n",
    "                    hspace=.01)\n",
    "\n",
    "plot_num = 1\n",
    "\n",
    "default_base = {'quantile': .3,\n",
    "                'eps': .3,\n",
    "                'damping': .9,\n",
    "                'preference': -200,\n",
    "                'n_neighbors': 10,\n",
    "                'n_clusters': 3,\n",
    "                'min_samples': 20,\n",
    "                'xi': 0.05,\n",
    "                'min_cluster_size': 0.1}\n",
    "\n",
    "datasets = [\n",
    "    (noisy_circles, {'damping': .77, 'preference': -240,\n",
    "                     'quantile': .2, 'n_clusters': 2,\n",
    "                     'min_samples': 20, 'xi': 0.25}),\n",
    "    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),\n",
    "    (varied, {'eps': .18, 'n_neighbors': 2,\n",
    "              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),\n",
    "    (aniso, {'eps': .15, 'n_neighbors': 2,\n",
    "             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),\n",
    "    (blobs, {}),\n",
    "    (no_structure, {})]\n",
    "\n",
    "for i_dataset, (dataset, algo_params) in enumerate(datasets):\n",
    "    # update parameters with dataset-specific values\n",
    "    params = default_base.copy()\n",
    "    params.update(algo_params)\n",
    "\n",
    "    X, y = dataset\n",
    "\n",
    "    # normalize dataset for easier parameter selection\n",
    "    X = StandardScaler().fit_transform(X)\n",
    "\n",
    "    # estimate bandwidth for mean shift\n",
    "    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])\n",
    "\n",
    "    # connectivity matrix for structured Ward\n",
    "    connectivity = kneighbors_graph(\n",
    "        X, n_neighbors=params['n_neighbors'], include_self=False)\n",
    "    # make connectivity symmetric\n",
    "    connectivity = 0.5 * (connectivity + connectivity.T)\n",
    "\n",
    "    # ============\n",
    "    # Create cluster objects\n",
    "    # ============\n",
    "    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)\n",
    "    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])\n",
    "    ward = cluster.AgglomerativeClustering(\n",
    "        n_clusters=params['n_clusters'], linkage='ward',\n",
    "        connectivity=connectivity)\n",
    "    spectral = cluster.SpectralClustering(\n",
    "        n_clusters=params['n_clusters'], eigen_solver='arpack',\n",
    "        affinity=\"nearest_neighbors\")\n",
    "    dbscan = cluster.DBSCAN(eps=params['eps'])\n",
    "    optics = cluster.OPTICS(min_samples=params['min_samples'],\n",
    "                            xi=params['xi'],\n",
    "                            min_cluster_size=params['min_cluster_size'])\n",
    "    affinity_propagation = cluster.AffinityPropagation(\n",
    "        damping=params['damping'], preference=params['preference'])\n",
    "    average_linkage = cluster.AgglomerativeClustering(\n",
    "        linkage=\"average\", affinity=\"cityblock\",\n",
    "        n_clusters=params['n_clusters'], connectivity=connectivity)\n",
    "    birch = cluster.Birch(n_clusters=params['n_clusters'])\n",
    "    gmm = mixture.GaussianMixture(\n",
    "        n_components=params['n_clusters'], covariance_type='full')\n",
    "\n",
    "    clustering_algorithms = (\n",
    "        ('MiniBatchKMeans', two_means),\n",
    "        ('AffinityPropagation', affinity_propagation),\n",
    "        ('MeanShift', ms),\n",
    "        ('SpectralClustering', spectral),\n",
    "        ('Ward', ward),\n",
    "        ('AgglomerativeClustering', average_linkage),\n",
    "        ('DBSCAN', dbscan),\n",
    "        ('OPTICS', optics),\n",
    "        ('Birch', birch),\n",
    "        ('GaussianMixture', gmm)\n",
    "    )\n",
    "\n",
    "    for name, algorithm in clustering_algorithms:\n",
    "        t0 = time.time()\n",
    "\n",
    "        # catch warnings related to kneighbors_graph\n",
    "        with warnings.catch_warnings():\n",
    "            warnings.filterwarnings(\n",
    "                \"ignore\",\n",
    "                message=\"the number of connected components of the \" +\n",
    "                \"connectivity matrix is [0-9]{1,2}\" +\n",
    "                \" > 1. Completing it to avoid stopping the tree early.\",\n",
    "                category=UserWarning)\n",
    "            warnings.filterwarnings(\n",
    "                \"ignore\",\n",
    "                message=\"Graph is not fully connected, spectral embedding\" +\n",
    "                \" may not work as expected.\",\n",
    "                category=UserWarning)\n",
    "            algorithm.fit(X)\n",
    "\n",
    "        t1 = time.time()\n",
    "        if hasattr(algorithm, 'labels_'):\n",
    "            y_pred = algorithm.labels_.astype(np.int)\n",
    "        else:\n",
    "            y_pred = algorithm.predict(X)\n",
    "\n",
    "        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)\n",
    "        if i_dataset == 0:\n",
    "            plt.title(name, size=18)\n",
    "\n",
    "        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',\n",
    "                                             '#f781bf', '#a65628', '#984ea3',\n",
    "                                             '#999999', '#e41a1c', '#dede00']),\n",
    "                                      int(max(y_pred) + 1))))\n",
    "        # add black color for outliers (if any)\n",
    "        colors = np.append(colors, [\"#000000\"])\n",
    "        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])\n",
    "\n",
    "        plt.xlim(-2.5, 2.5)\n",
    "        plt.ylim(-2.5, 2.5)\n",
    "        plt.xticks(())\n",
    "        plt.yticks(())\n",
    "        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),\n",
    "                 transform=plt.gca().transAxes, size=15,\n",
    "                 horizontalalignment='right')\n",
    "        plot_num += 1\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.savefig('img.png')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# with grid search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dataset read\n",
      "dataset read complete\n",
      "dataset column drop complete\n",
      "algo started\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.7/site-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.\n",
      "  warnings.warn(CV_WARNING, FutureWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DBSCAN(algorithm='auto', eps=0.5, leaf_size=20.0, metric='euclidean',\n",
      "       metric_params=None, min_samples=1000, n_jobs=None, p=None)\n",
      "algo finished\n",
      "time return result\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th>serial_number</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>model</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST12000NM0007</th>\n",
       "      <th>-1</th>\n",
       "      <td>2952</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>35304</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST4000DM000</th>\n",
       "      <th>-1</th>\n",
       "      <td>1921</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>21473</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST8000DM002</th>\n",
       "      <th>-1</th>\n",
       "      <td>769</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>9167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST8000NM0055</th>\n",
       "      <th>-1</th>\n",
       "      <td>1220</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13459</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      serial_number\n",
       "model         labels               \n",
       "ST12000NM0007 -1               2952\n",
       "               0              35304\n",
       "ST4000DM000   -1               1921\n",
       "               1              21473\n",
       "ST8000DM002   -1                769\n",
       "               2               9167\n",
       "ST8000NM0055  -1               1220\n",
       "               2              13459"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "from sklearn.metrics import silhouette_score as sc\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.cluster import DBSCAN\n",
    "import datetime\n",
    "\n",
    "def cv_silhouette_scorer(estimator, X):\n",
    "    estimator.fit(X)\n",
    "    cluster_labels = estimator.labels_\n",
    "    num_labels = len(set(cluster_labels))\n",
    "    num_samples = len(X.index)\n",
    "    if num_labels == 1 or num_labels == num_samples:\n",
    "        return -1\n",
    "    else:\n",
    "        return sc(X, cluster_labels)\n",
    "\n",
    "def dbscan():\n",
    "    st_models = req_models = ['ST12000NM0007', 'ST4000DM000', 'ST8000NM0055', 'ST8000DM002']\n",
    "    df_final = pd.DataFrame()\n",
    "    print(\"dataset read\")\n",
    "    for name in st_models:\n",
    "        full_name = name + \"_last_day_all_q_raw.csv\"\n",
    "        dataset = pd.read_csv(\"../dataset/\" + full_name)\n",
    "        df = dataset[['model', 'serial_number','capacity_bytes', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']]\n",
    "        df_final = df_final.append(df, ignore_index=True)\n",
    "    print(\"dataset read complete\")    \n",
    "    df_final_copy = df_final.copy()\n",
    "    drop_columns = ['model', 'serial_number']\n",
    "    df_final_copy.drop(drop_columns, axis = 1, inplace=True)    \n",
    "    print(\"dataset column drop complete\")\n",
    "    print(\"algo started\")\n",
    "    \n",
    "    #####\n",
    "    param_dict = {}\n",
    "    param_dict[\"min_samples\"] =  [500, 1000, 1500, 2000]\n",
    "    param_dict[\"leaf_size\"] = np.linspace(20, 100, 5)\n",
    "    gs = GridSearchCV(estimator=DBSCAN(), param_grid=param_dict, \n",
    "                  scoring=cv_silhouette_scorer, n_jobs=4)\n",
    "    gs.fit(df_final_copy)\n",
    "    print(gs.best_estimator_)\n",
    "    #####\n",
    "    db = gs.best_estimator_ #DBSCAN( min_samples=1000)\n",
    "    db.fit(df_final_copy)\n",
    "    print(\"algo finished\")\n",
    "    labels = db.labels_\n",
    "    df_new = pd.DataFrame()\n",
    "    df_new[\"model\"] = df_final[\"model\"]\n",
    "    df_new[\"serial_number\"] = df_final[\"serial_number\"] \n",
    "    df_new[\"labels\"] = labels\n",
    "    dx = df_new.groupby(['model', 'labels']).count()\n",
    "    print(\"time return result\")\n",
    "    return dx\n",
    "result = dbscan()    \n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'df_new' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-3-94dbd581444b>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mdf_new\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m: name 'df_new' is not defined"
     ]
    }
   ],
   "source": [
    "\n"
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
   "display_name": "Python 3.7.4 64-bit",
   "language": "python",
   "name": "python37464bit8838e21937b04b589d195beda5a2136d"
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
