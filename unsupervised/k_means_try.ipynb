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
    "# from time import time\n",
    "from sklearn.cluster import KMeans\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import preprocessing\n",
    "import matplotlib.patches as mpatches"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## K-Means for one model\n",
    "\n",
    "Tryring to see if it can cluster based on failure index. No good clustering. Majority of the failed ones fall in the same class as majority of good ones.\n",
    "\n",
    "Tried for ST12000NM0007_last_day_all - raw, normalised, upsample raw upsample normalised, downsample raw, down sampe normalised\n",
    "las_10_days - undersampled raw, oversampled raw"
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
       "(38256, 24)"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_csv(\"../dataset/ST12000NM0007_last_day_all_q_raw.csv\")\n",
    "dataset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "smart_attributes=['smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "d_copy=dataset.copy() \n",
    "d_copy.shape\n",
    "d_5f=d_copy[smart_attributes] #get only smart attributes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def normalize(input_frame):\n",
    "    #since features are vearied by differnt scales, we normalize\n",
    "    # scaler= preprocessing.StandardScaler()\n",
    "    scaler= preprocessing.MinMaxScaler()\n",
    "    scaler.fit(input_frame)\n",
    "    data_transformed = scaler.transform(input_frame)\n",
    "    return data_transformed\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_elbow(input_frame,plot_title=\"Elbow Method For Optimal k\" ):\n",
    "    Sum_of_squared_distances = []\n",
    "    K = range(1,15)\n",
    "    for k in K:\n",
    "        km = KMeans(n_clusters=k)\n",
    "        km = km.fit(input_frame)\n",
    "        Sum_of_squared_distances.append(km.inertia_)\n",
    "    plt.plot(K, Sum_of_squared_distances, 'bx-')\n",
    "    plt.xlabel('k')\n",
    "    plt.ylabel('Sum_of_squared_distances')\n",
    "    plt.title(plot_title)\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_kmeans(input_frame,k=2,random_state=None, n_init=10):\n",
    "    kmeans = KMeans(n_clusters=k,random_state=random_state, n_init=n_init)\n",
    "    output=kmeans.fit(input_frame)\n",
    "    labels=output.labels_\n",
    "#     np.unique(labels, return_counts=True)\n",
    "    return labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def display_result_frame(input_frame,labels):\n",
    "    df_new = pd.DataFrame()\n",
    "    # df_new[\"model\"] = d_copy[\"model\"]\n",
    "    df_new[\"serial_number\"] = input_frame[\"serial_number\"] \n",
    "    df_new[\"labels\"] = labels\n",
    "    df_new[\"failure\"]=input_frame[\"failure\"]\n",
    "    return df_new.groupby(['failure', 'labels']).count()\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
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
       "      <th>failure</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"6\" valign=\"top\">0</th>\n",
       "      <th>0</th>\n",
       "      <td>36961</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"6\" valign=\"top\">1</th>\n",
       "      <th>0</th>\n",
       "      <td>837</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>80</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>162</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>45</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                serial_number\n",
       "failure labels               \n",
       "0       0               36961\n",
       "        1                  29\n",
       "        2                  77\n",
       "        3                  44\n",
       "        4                   2\n",
       "        5                   9\n",
       "1       0                 837\n",
       "        1                  80\n",
       "        2                 162\n",
       "        3                   5\n",
       "        4                   5\n",
       "        5                  45"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_transformed=normalize(d_5f)\n",
    "plot_elbow(data_transformed,'Elbow Method For Optimal k - only 5 features -with normalization')\n",
    "clustered_labels=run_kmeans(data_transformed,6)\n",
    "display_result_frame(d_copy,clustered_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Majority of failed points getting clustered to the cluster in which majority of the good ones present. Not a good clustering."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Without normalization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
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
       "      <th>failure</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"6\" valign=\"top\">0</th>\n",
       "      <th>0</th>\n",
       "      <td>36746</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>276</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>34</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>56</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"5\" valign=\"top\">1</th>\n",
       "      <th>0</th>\n",
       "      <td>1006</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>101</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>22</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                serial_number\n",
       "failure labels               \n",
       "0       0               36746\n",
       "        1                   9\n",
       "        2                   1\n",
       "        3                 276\n",
       "        4                  34\n",
       "        5                  56\n",
       "1       0                1006\n",
       "        2                   2\n",
       "        3                 101\n",
       "        4                   3\n",
       "        5                  22"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "plot_elbow(d_5f,'Elbow Method For Optimal k - only 5 features -without normalization')\n",
    "clustered_labels=run_kmeans(d_5f,6,random_state=10,n_init=15)\n",
    "display_result_frame(d_copy,clustered_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Not a big difference than the normalised one :("
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Undersampling\n",
    "\n",
    "#### Random under sampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Good entry count:  37122 \n",
      "Failed entry count:  1134\n"
     ]
    }
   ],
   "source": [
    "d_copy.groupby(['failure']).count()['serial_number']\n",
    "count_good, count_bad = d_copy.failure.value_counts()\n",
    "print(\"Good entry count: \",count_good,\"\\nFailed entry count: \", count_bad)\n",
    "\n",
    "# Divide by class\n",
    "df_good_all = d_copy[d_copy['failure'] == 0]\n",
    "df_bad_all = d_copy[d_copy['failure'] == 1]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random under-sampling:\n",
      "1    1134\n",
      "0    1134\n",
      "Name: failure, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "df_good_under = df_good_all.sample(count_bad)\n",
    "df_test_under = pd.concat([df_good_under, df_bad_all], axis=0)\n",
    "\n",
    "print('Random under-sampling:')\n",
    "print(df_test_under.failure.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_under_5f=df_test_under[smart_attributes]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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dLvlgQOLjtjMmslJQe9xUtDLnBT03MKF6XbKU6bAeefBzJnLhq2+emoKuc02sPXW6bXFFqmUwcz6JycFvceHSut31lgD3v3uVCJwxhnw9a/DL34Bn/oULFqULjecfXa6eRKkyopvf/uyRGGbbeAtb0l1FszMbBknBdYvVdYh2H33Zd2nnAJLlsCDD6YEYdq09D5pEpx+epp+lVXgbW9bliRsvXXqHjLElRjNrHP58kEv8+WD5liZP+6lS2HGjGVJQvn9uefS8EGDUgnCxhun1hAnnABHHgm33eZKjGZF8uWD3uOkoJc5KehbIuDRR5dPEqZNg2efXTaOBBttBKNHw/DhsMEG6T3/yvfr6gZMLokw6x4nBb3Hlw+so0nplsybbgof/GDqFwGzZsExx6QmjxMmwLhxMGcOPPII/POfKWmo9XjooUNrJw/Dh6f5779/upSx994pGTj44FQS0VOtTjic0Ji1NycFNUjaCzgFGAj8OiJOqBi+KnA+sA0wFzgwIh7t7Tit+SR46KH0Z/fNb6Y/7xNPXP6PMAKefz4lCrNnp/fK1+zZ8PjjqeRhzpwVn/Vw8MHLPg8eDAccAGuumSpSrrFG7c/1hg0blubzq1/BnnumVhqHHNKchANSQpC/lJKv29FTrUw4+nOy5NitNzkpqELSQODnwB7ALOB2SVdExP250T4OPBcRJUkHAScCB/Z+tNZslZUYd911xToFUnqmw7rrwhvf2PU8I9L9FvJJxK9/DVdemW7GtO22qanlwoXw4ovLPj/11PL9X3yxdglFXrnUo+w970mVK8uvwYOX767Xv7LfbrvB+96XSlCmTYMDD0z1LqZOTS1CBg1K7+VXvrvW54ED0/f813/B974H73hHupTzta+luh133rnsvhTSss+Ndr/xjfChD8E558BOO6WbY330o3D++fDSS8umyb+XX41oZbLUynm3ev6tjt2az3UKqpC0PfCdiHhP1n0cQET8KDfOX7JxbpE0CHgaGB5drFDXKej7euPspnxw/MxnUklEdyoxvvpq9QQi//nii+HPf04PotpppzTNa6+l9/yrWr9Gxn3ppZTodILKRKFa8jBgQGrxsmhRqlPyyivpMtLgwcsnF/n37vR7+eV0yWro0PQ48uHDYbXVlk9aKhOY7gxbtCgloOusk0rANt443fOj2rroal1VWrgQnngCRo5M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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#without normalization\n",
    "plot_elbow(df_under_5f,'Elbow Method For Optimal k - only 5 features -without normalization undersampled')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
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
       "      <th>failure</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">0</th>\n",
       "      <th>0</th>\n",
       "      <td>1132</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"4\" valign=\"top\">1</th>\n",
       "      <th>0</th>\n",
       "      <td>1107</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                serial_number\n",
       "failure labels               \n",
       "0       0                1132\n",
       "        2                   2\n",
       "1       0                1107\n",
       "        1                   1\n",
       "        2                  25\n",
       "        3                   1"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clustered_labels=run_kmeans(df_under_5f,4,random_state=10)\n",
    "display_result_frame(df_test_under,clustered_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgIAAAEWCAYAAAAU6v/cAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nO3deZgcZbn+8e+dBAgkARIStoQQNllVliSD4EHDJmBYVFQUMQgKsitK1HP0d/QoLshBQRFFUFYRBIWAImAcwiJIEjbZJBACiSxJICEQCRB4fn+81Wc6ne6enslM18zU/bmuurr2frqquvrpt96qVxGBmZmZFVO/vAMwMzOz/DgRMDMzKzAnAmZmZgXmRMDMzKzAnAiYmZkVmBMBMzOzAlulREDSkZLuKBsOSVuuelj568rPImmOpL27Yl09gaRXJW3eDetd4XiqmDYm2ycDuvp9O6KZcUj6jqSFkp7v7vfqDdo77vrK90zS+yXNKxt+WNL7u+F9uuV73Ew95bzQlZq1/8u1mwhkX67XsoOm1P20O4NqVPbDEZLOqhh/SDb+ogbXc6ukz3ZLkO2/90WS3qjYvh/vwvVPlHSPpKWSXpR0uaRRHVh+pW0TEYMjYnZXxVgUNfZ1/xrzbgJ8CdguIjZcxfdd4cTSW5Ufd9m2/E7eMTVDRGwfEbeuyjr8Pe69umL/t6fREoEDs4Om1J3YnUF10JPAxysywk8Dj+cUT2ecUbF9r+zoCqr9oEg6FPgNcDYwHNgeeB24Q9LQVQ3aOqVyX79VY75NgRcjYn4zg6umL/3b6g7ePr2T91ub7qgjcICk2VmR5g8l9QOQ1E/S1yU9LWm+pEskrZNNu1jSl7L+kdm/+eOz4S0lvSRJNd7veeAfwAey+YcBuwFTymeStKukv0laLOmBUlGLpNOB/wB+WqW0Y29JsyQtknRuKYZ6nyWbfkQ27UVJ/9XZDSlp2yyTX5wVDx1UNu0iSedJ+pOkpcCEimUF/C/wnYi4PCJei4jngc8CrwJfzOY7UtKdkn4i6WVJj0naq962UdllkyyOn0m6MZvnTkkbSvpxtt0ek7RTWVxflfSkpFckPSLpQ53cNh9RKq3aoZPL7yZpevaZp0varWzarZK+nX2WVyTdLGl4lXV8VNLMinFfknRtZ2IqW8fewC3Axtk2vSgbX/UYzqZ9RtKjWbyzJR2bjR8E3Fi2rlclbayKf9RauThyjqSvSHoQWCppQLbcNZIWSHpK0sll84+XNEPSEkkvqKKUrs5n/Yyk68uGn5B0VdnwXEk7Zv2hdD44BjgcmJx9nuvLVrmjpAez/XqlpIE13vdISXdIOjM7Tp+StH/Z9I0lTVE69zwh6XNl074p6WpJl0laAhyZjftdNu4VSf+Q9A5JX1M6R8yVtG97+6tGrP93ySPb96X9uDTbJmMkDZV0Q7ZvFmX9o7JlGvker6N0HlugdO76utrO3XW3VUeponRC1S8xf17Vz739szgWSpoNfLBi3etIulDSc5L+pXR5rX/Z+9wp6UeSXgK+WSW2msdxtn+fz46t2yRtXzato+fBOdmx8Ug2/dd1jtXy/f9NSVdl++oVpd+FsWXz7izpvmza77LvQPslZxFRtwPmAHvXmHYkcEfZcACtwDBgNOlf+WezaUcBTwCbA4OB3wOXlk27Puv/JOlf/pVl066r9/7ZMqX5jwd+AXwHuCgbNxJ4ETiAlPzskw2PyKbfWoqz4rPcAKybfZYFwH4NfJbtSD+0ewBrAGcBy+tsw4tIP9aV41fL3uM/gdWBPYFXgK3LlnsZ2D37TAMrlt8m+wybVVn3t4C7yrbhclJisBrw8Wy9w9rZNluWxbEQ2AUYCPwVeIpUKtM/2w+tZct+FNg4i/njwFJgo2rHU8V7jsnedwDwmWzbbNne8VtjXcOARcAR2fo+kQ2vV/aZnwTeAayZDX+/ShxrAC8B25at+z7gI3X29UtZN7PWfNm87wfmlQ23dwx/ENgCEPA+4N/AztXWVe24q/J+c4D7gU2ybdAvi/n/kY7HzYHZwAey+e8Cjsj6BwO7NrgvNgcWZ+vfCHga+FfZtEVAvxrH3Xcq1jUHuCc7voYBjwKfr3PueBP4HOk4PQ54FlA2fRrwM9IxvSPp+79XNu2b2bKHZHGvmY1bRvpDMgC4hPQ9+C/S9+pzwFNl79/w/qLGORj4LnBbtv71gI8AawFDgN8B15bNeyv1v8eXANdly44hnbuPbmRbdeL7t0IsVP8dqXXu/TzwGOm4HEb6vQlgQDb9WtL5fxCwfnY8HFtxrjsp20drVomt5nFMOu8PIX3vfwzcX/F96sh5cA7wUNnnuJPseK63/2k7zg7I1vs94O5s2uqk788p2THxYeANqvy+rPS5G9hpc0g/bIvLus/V2YH7lQ0fD0zN+qcCx5dN2zo7uAaQvhClk8HPgWNLGwK4GDi1zpf5DtIX8QVgHeBu0o9jeSLwFbIf6rJlbwImtfMleW/Z8FXAVxv4LP8P+G3ZtEHZzqiXCCwr27YLs/H/QSrt6Fc27xXAN8uWu6TOfntv9hkGVpn2eWBW2TZc4UtN+vKUvgy1tk35CfmXZdNOAh4tG34nsLhOnPcDB1c7nirmG5O975eBR4BRnTkJZes6ArinygngyLLP/PWK4/jPFXGUTjznAadn/duTfrjWqPG+O5NO2ANIX+RXgN1rzPt+VjwZ1D2Gqyx/LXBKtXWV7bf2EoGjyoZbgGcq1vE14NdZ/22kBHN4J/bH3GzbHAacnx1/25ASvil1jrtqicCnyobPAH5e4z2PBJ4oG14rW/+GpJPzW8CQsunfo+188k3gtor1fRO4pWz4QNJ5s382PCRb/7od3V9USQRISfQcskSwyvp2BBaVDd9Kje8x6QfldVJ9lNK0Y4Fb29tWnfz+rRAL1X9Hap17/0pZcgfsS1tivkH2OdYsm/4Jsh/g7H2eaSe2ho5jUpISwDplx2PD58Fs35V/jgOAJ9vb/9lx9peyadsBr2X9ewD/YsVz+R00kAg0emngkIhYt6z7ZZ1555b1P03Kzslen66YNgDYICKeJH1pdiT9AN4APCtpa1K2PK1ecBHxGvBH4OukHXhnxSybAh/NitUWS1pM+qHcqN56ST/EJf8mZYh1P0s27f+2QUQsJf1zq+fMsm1bKoLeGJgbEW9XvM/IsuHybV1pYfZa7TNuVDYd0j+wqHifjWncC2X9r1UZLm03JH1a0v1l+2EHUv2FRp0GnBsRNSu/lRXPvSrp8CqzVO4/WHnb1tr3lS4GPpkVXR4BXBURr1ebMSLujYgXI2J5RPwJuJyUtTei7jEsaX9JdysVZS8mnVg6sl2rKT++NiVdXih///8kHfMAR5NKUB5TutQysdoKa+ybaaST3x5Z/62k73273/0qGt1vK8wbEf/OegeTjo+XIuKVsnkb+e5VHvcLo60OyGtl61+l/ZUVMf8U+FBELMjGrSXpF1mx/hLSD9q6qlEZtcJw2v5NltT8PlRsq8rYDi/bvzc28nlqqHfurfyNKdmU9E/4ubJj9BekkoGSeudMqHEcZ5ckvq90WXMJ6ccZVtxnDZ8Hq8TSkXNu5bYZqFTfYWNWPpe393mB9OPV1TYBHs76R5P+bZK9blo232hSMU1pY00DDgVWj4h/SZpGKlYZSvrX2J5LSNnit6pMm0v6N/W5KtMgZXYdUe+zPAdsW5ogaS3Sv8COehbYRFK/smSgdLmlpF7c/wTmkYrizyiLpx+pCLH8OvZISSo7gEbTVseio9umJkmbAr8E9iJdmnhL0v2k4tFG7Qv8WdLzEXFNtRkior3rl5X7D9Jn/nMH4ii9192S3iAlsJ/MuoYXp/HPXvMYlrQGcA3p+3JdRLypVE+htO5q+3Ap6Z9dSbU7EypPKE9FxFbVgouIWcAnsuPrw8DVktbLEuHy+artm2mkf9CbkYq7F5PqALyH9INX9S1rjO8KzwLDJA0pSwZGk/5trfL7N7C/6i07AvgDcGJE3Fc26UukksmWiHheqV7FfdQ/BkoWkko0NyWVtsHKn7chEXE5KcGtp5Fjr5bnSL8xJaPL+ueSSgSGR8TyWiHWW3mt4zjrPxjYm5QErEMq/evIuatS5ed4ttaMDXqOlc/lm5Auc9bVHZUFT1OquLIJ6VpFqQb8FcAXJW0maTDpC39l2Q6bBpxIymQh/Ss4iVRkVKtmdblppOumP6ky7TLgQEkfyDK7gUqVo0q30b1Auh7ZqHqf5WpgoqT3Slod+B86t53/TvrCTJa0mlLFsAOB3zaycHYgfBn4uqRPSlpT0obABcDawI/KZl8fODl7n4+SEpk/ZdM6um3qGUT6Ipb+xXyGVCLQEQ8D+wHnqqzyZAf9CXhHtl0GKN2uuR2pJKozLiH9YC2PiKrPQYB0F4ekwUqVTfcFPkVFpdY66h3Dq5OuWy4AlitV5Nq3bNkXgPVUVqGVlFwfIGlYdlx8oZ33vwdYolSBcM0shh0kjcs+26ckjciS1sXZMo18byF9dyeQinTnAbeT9vF6pB+zarryuFxBRMwF/gZ8L9vO7yL9U2zvB65R7e2vqrJ/fdcAl8fKdxYNIf3rXKxUYfq/K6bX3F7Z+fUq4HRJQ7KE/VTSMdcd7gc+nJVibEnato26inSuGqV059NXSxMi4jngZuB/Ja2dfc+2kPS+Rlde5zgeQkoyXiQlMd/tQMy1nJB9jmGk0rUO3y1W4S5SrCdm57WDgfGNLNjoD9T1WvHe5z/Umfc6UqWi+0nF9Rdm438FXEr6oX+KdF38pLLlppE2dikRuIO0wW+jAZFMjYiXqkybS8rm/pP05ZtLKmIuff6zgUOVam+e08Db1fwsEfEwcALptr3nSFljh+/hjog3gIOA/UkZ+8+AT0fEYx1Yx5Wk4uovZut4hFSfYveIKL9c8Xdgq2ye04FDy6Z3dNvUi+cR0p0Md5FOTO8kVZLp6HoeACYCv1Qnai9nn20i6V/Ui8BkYGJELKy7YG2XkhKaS9uZ7xTSv6zFwA9JdW1ubTDmmsdw9q/1ZNJJchGpVGJK2bKPkZLX2VmR6cZZrA+Q/t3cTDsnoezH4kDS5bunSMfKBaR/RpB+uB+W9CrpmDksIpY1+NkeJ10avD0bXkKqiHhnnT8BFwLbZZ9nle7SqOETpPogz5L+gf93RNzSFStub3/VMYpU8vSFivPxaFLltTVJ++VuVi7dau97fBLpj8ds0rn3N6TzXHf4Eane1AukS2sdSbB+Saob8wBwL6midrlPkxKtR0jb9mravwRcrtZxfAlZRdZs3Xd3YJ21/Ib03Zuddav0XIzsN+PDpMRqMemPxg2kBKauUg1ZKyhJR5Iq7rw371h6K0lrAvNJtb5n5R2PmfVskuaQzrt/6eb3+Tupwuyv683ntgbMVt1xwHQnAWaWJ0nvU3p+wQBJk4B30UDdJz9ZyWwVZJm9SPeUm5nlaWvSJafBpEqCh2Z1J+rypQEzM7MC86UBMzOzAvOlgSYZPnx4jBkzJu8wzMx6jZkzZy6MiBF5x9HXORFokjFjxjBjxoy8wzAz6zUkVT4B1LqBLw2YmZkVmBMBMzOzAnMiYGZmVmBOBMzMzArMiYCZmVmBORHooc44A1pbVxzX2prGm5mZdRUnAj3UuHHwsY+1JQOtrWl43Lh84zIzs77FzxHooSZMgCuugAMPTN1f/gJXXZXGm5mZdRWXCPRge+8N/fvDb38Lxx3nJMDMzLqeE4EerLUVli2DddeF885buc6AmZnZqnIi0A5JcyT9Q9L9kmZk44ZJukXSrOx1aFe/b6lOwJFHwuLFcP75K9YZMDMz6wpOBBozISJ2jIix2fBXgakRsRUwNRvuUtOnpzoBhx+ehldfPQ1Pn97V72RmZkXmyoKdczDw/qz/YuBW4Ctd+QaTJ6fXpUtTPYG//x3+539cT8DMzLqWSwTaF8DNkmZKOiYbt0FEPAeQva5fbUFJx0iaIWnGggULOvXmgwbBDjukRMDMzKyrORFo3+4RsTOwP3CCpD0aXTAizo+IsRExdsSIzjepPX483HMPvP12p1dhZmZWlROBdkTEs9nrfOAPwHjgBUkbAWSv87szhpaWVGHwiSe6813MzKyInAjUIWmQpCGlfmBf4CFgCjApm20ScF13xtHSkl59ecDMzLqaE4H6NgDukPQAcA/wx4j4M/B9YB9Js4B9suFus+22MHiwEwEzM+t6vmugjoiYDby7yvgXgb2aFUf//qmNAScCZmbW1Vwi0EuMHw8PPJCeNGhmZtZVnAj0Ei0t8OabcP/9eUdiZmZ9iROBXsIVBs3MrDs4EeglNt4YRo1yImBmZl3LiUAv0tLiRMDMzLqWE4FeZPx4mD0bOvm0YjMzs5U4EehFSvUE3AKhmZl1FScCvcguu0C/fr48YGZmXceJQC8yeLBbIjQzs67lRKCXaWlJLRFG5B2JmZn1BU4Eepnx42HRIpg1K+9IzMysL3Ai0MuUKgzec0++cZiZWd/gRKCX2W47t0RoZmZdx4lAL9O/P4wd60TAzMy6hhOBXqilJTU+5JYIzcxsVTkR6IXGj3dLhGZm1jWcCPRCrjBoZmZdxYlALzRyZOpcT8DMzFaVE4Feyi0RmplZV3Ai0Eu1tMCTT8LChXlHYmZmvZkTgV5q/Pj06noCZma2KpwI9FJjx6aWCJ0ImJnZqnAi0EsNHgzbb+96AmZmtmqcCPRibonQzMxWVWESAUmnSFpbyYWS7pW0b95xrYqWFnjpJXjiibwjMTOz3qowiQBwVEQsAfYFRgCfAb6fb0irplRh0JcHzMyss4qUCCh7PQD4dUQ8UDauV9p+exg0yBUGzcys84qUCMyUdDMpEbhJ0hDg7ZxjWiVuidDMzFZVkRKBo4GvAuMi4t/A6qTLA71aqSXC11/POxIzM+uNipQIBLAdcHI2PAgYmF84XaOlBd54wy0RmplZ5xQpEfgZ8B7gE9nwK8C5+YXTNVxh0MzMVkWREoGWiDgBWAYQEYtIlwfaJam/pPsk3ZANbybp75JmSbpSUkPr6Q6jRsHGG7vCoJmZdU6REoE3JfUnXSJA0ggaryx4CvBo2fAPgB9FxFbAIlL9g9y4JUIzM+usIiUC5wB/ANaXdDpwB/Dd9haSNAr4IHBBNixgT+DqbJaLgUO6I+BGtbSkhwq9+GKeUZiZWW80IO8AmiUiLpc0E9iL9PyAQyLi0XYWA/gxMBkYkg2vByyOiOXZ8DxgZLUFJR0DHAMwevToVYi+vpaW9HrPPbD//t32NmZm1gcVpkRA0q7AvyLi3Ij4KTBPUks7y0wE5kfEzPLRVWat+rT/iDg/IsZGxNgRI0Z0Ovb27LILSL48YGZmHVeYRAA4D3i1bHhpNq6e3YGDJM0Bfku6JPBjYF1JpdKUUcCzXRtqxwwZkp4y6AqDZmbWUUVKBBTR1k5fRLxNO5dGIuJrETEqIsYAhwF/jYjDgVbg0Gy2ScB13RNy49wSoZmZdUaREoHZkk6WtFrWnQLM7uS6vgKcKukJUp2BC7ssyk5qaUmVBZ98Mu9IzMysNylSIvB5YDfgX6QKfi1kFfkaERG3RsTErH92RIyPiC0j4qMRkfsDfksVBl1PwMzMOqIwiUBEzI+IwyJi/YjYICI+GRHz846rq2y3Hay1lhMBMzPrmMLcPpg9QOhzwBjKPndEHJVXTF1pwIDUEqErDJqZWUcUJhEgVei7HfgL8FbOsXSLlhY4++zUEuEaa+QdjZmZ9QZFSgTWioiv5B1Edyq1RPjAA22NEZmZmdVTmDoCwA2SDsg7iO7kCoNmZtZRRUoETiElA69JWiLpFUlL8g6qK40cCRtt5ETAzMwaV5hLAxExpP25ejep7cFCZmZmjShMIgAgaSiwFTCwNC4ibssvoq7X0gLXXgsvvQTDhuUdjZmZ9XSFuTQg6bPAbcBNwLey12/mGVN3KG+J0MzMrD2FSQRIdQTGAU9HxARgJ2BBviF1vbFj3RKhmZk1rkiJwLKIWAYgaY2IeAzYOueYutyQIekpg04EzMysEUWqIzBP0rrAtcAtkhaRc/PB3aWlBa67LrVEKOUdjZmZ9WSFKRGIiA9FxOKI+CbwDVKLgQfnG1X3KLVEOLuzbSuamVlhFCYRkHRpqT8ipkXEFOBXOYbUbfxgITMza1RhEgFg+/IBSf2BXXKKpVttv71bIjQzs8b0+URA0tckvQK8K3ui4JJseD6pIaI+Z8AA2GUXJwJmZta+Pp8IRMT3sqcK/jAi1s66IRGxXkR8Le/4uktLC9x3X2qEyMzMrJY+nwiUuUHSIABJn5J0lqRN8w6qu5S3RGhmZlZLkRKB84B/S3o3MBl4Grgk35C6jysMmplZI4qUCCyPiCDdMnh2RJwN9NmGiEaNckuEZmbWviI9UOgVSV8DPgXskd01sFrOMXUbCcaPdyJgZmb1FalE4OPA68DREfE8MBL4Yb4hda+WFpg1K7VEaGZmVk1hEoGIeD4izoqI27PhZyKiz9YRgLZ6AtOn5xuHmZn1XH0+EZB0R/b6StlzBJaUhvOOrzu5JUIzM2tPn68jEBHvzV77bMXAWtZe2y0RmplZfX0+EZA0rN70iOjTV9DHj4cpU9wSoZmZVdfnLw0AM4EZ2esC4HFgVtY/M8e4mqLUEuFTT+UdiZmZ9UR9PhGIiM0iYnPgJuDAiBgeEesBE4Hf5xtd9/ODhczMrJ4+nwiUGRcRfyoNRMSNwPtyjKcpdtjBLRGamVltfb6OQJmFkr4OXAYE6cFCL+YbUvdzS4RmZlZPkUoEPgGMAP6QdSOycTVJGijpHkkPSHpY0rey8ZtJ+rukWZKulLR6t0e/CsaPd0uEZmZWXWESgYh4KSJOiYidImLniPhC+R0Dkn5SZbHXgT0j4t3AjsB+knYFfgD8KCK2AhYBRzfjM3RWSwu8/jo8+GDekZiZWU9TmESgAbtXjojk1WxwtawLYE/g6mz8xcAhTYmwk1xh0MzManEi0A5J/SXdD8wHbgGeBBZHxPJslnmkdgt6rE02gQ03dCJgZmYrcyLQjoh4KyJ2BEYB44Ftq81WbVlJx0iaIWnGggULujPMuqRUKuBEwMzMKjkRaFP3uXsRsRi4FdgVWFdS6Y6LUcCzNZY5PyLGRsTYESNGdGWsHdbSAo8/DosW5RqGmZn1ME4E2pxdOULSCEnrZv1rAnsDjwKtwKHZbJOA65oVZGeNH59e3RKhmZmV6/PPEZB0PTWK7gEi4qDs9aIqkzcCLpbUn5Q0XRURN0h6BPitpO8A9wEXdnngXWzcuLaWCPfdN+9ozMysp+jziQBwZvb6YWBD0gOFID1DYE69BSPiQWCnKuNnk+oL9Bprrw3bbut6AmZmtqI+nwhExDQASd+OiD3KJl0v6bacwspFSwtcf71bIjQzszZFqiMwQtLmpQFJm5GeLlgYLS2wcKFbIjQzszZ9vkSgzBeBWyXNzobHAMfmF07zlSoM3nMPbL55/XnNzKwYCpMIRMSfJW0FbJONeiwiXs8zpmZ75zthzTVTPYHDDss7GjMz6wkKc2lA0lrAacCJEfEAMFrSxJzDaiq3RGhmZpUKkwgAvwbeAN6TDc8DvpNfOPloaYF773VLhGZmlhQpEdgiIs4A3gSIiNdo52mCfZFbIjQzs3JFSgTeyJ4OGACStiA1M1wo5RUGzczMipQI/DfwZ2ATSZcDU4HJ+YbUfKNHwwYbuJ6AmZklhbhrQJKAx0hPF9yVdEnglIhYmGtgOXBLhGZmVq4QJQIREcC1EfFiRPwxIm4oYhJQ0tIC//ynWyI0M7OCJAKZuyWNyzuInqClJb26JUIzMytSIjABuEvSk5IelPQPSYWsOz92bLpE4AqDZmZWiDoCmf3zDqCnWGcd2GYb1xMwM7MClQhExNMR8TTwGukWwlJXSKUKg1HYLWBmZlCgREDSQZJmAU8B04A5wI25BpWjlhZYsADmzMk7EjMzy1NhEgHg26RbBx+PiM2AvYA78w0pP6UKg748YGZWbEVKBN6MiBeBfpL6RUQrsGPeQeXhjDNg4UIYOLAtEWhtTePNzKxYilRZcLGkwcBtwOWS5gPLc44pF+PGwcc+Bltske4caG1Nw1ddlXdkZmbWbIqC1BaTNAhYRnqq4OHAOsDlWSlBtxs7dmzMmDGjGW/VkNZW+OAHUwNEa68Nv/89TJiQd1RmZm0kzYyIsXnH0dcV5tJARCyNiLciYnlEXBwR5zQrCeiJJkyAY4+Ft9+G5cthq63yjsjMzPJQmERA0iuSlmTdMklvSVqSd1x5aW2Fyy6DY46BpUvhfe+DJYXdGmZmxVWYOgIRMaR8WNIhwPicwslVeZ2ACRNgyy1h8mTYc0+46y5YbbW8IzQzs2YpTIlApYi4Ftgz7zjyMH16WxIAcNppqZs5Ez7/eT9kyMysSApTIiDpw2WD/YCxFPTJgpMnrzzujDPS7YTf/jZsthl8/evNj8vMzJqvMIkAcGBZ/3LSkwUPzieUnulb30pPGvzGN2DMGPjUp/KOyMzMulthEoGI+EzeMfR0ElxwAcybB0cdBSNH+pZCM7O+rjCJgKRz6k2PiJObFUtPtvrqcM01sPvu8KEPwd/+Btttl3dUZmbWXYpUWXAgsDMwK+t2BN4CZmadZYYOhT/9KdUZOOAAeP75vCMyM7PuUpgSAWArYEJEvAkg6efAzRHxxXzD6pnGjIEbbkjPF5g4EaZNg0GD8o7KzMy6WpFKBDYGyp8lMDgbZzWMHQtXXgn33Qef+AS89VbeEZmZWVcrUiLwfeA+SRdJugi4F/huvQUkbSKpVdKjkh6WdEo2fpikWyTNyl6Hdn/4+Zg4EX7yE7j+ejjlFD9jwMysrylMIhARvwZagD9k3Xsi4uJ2FlsOfCkitgV2BU6QtB3wVWBqRGwFTM2G+6zjj4cvfxnOPRfOOivvaMzMrCsVJhGQtDvwSkRcR7pEMFnSpvWWiYjnIuLerP8V4FFgJOn5A6Uk4mLgkG4LvIf4wQ/gox9NCcHVV+cdjZmZdZXCJALAecC/Jb0bOA14Grik0YUljQF2Av4ObBARz0FKFoD1ayxzjKQZkmYsWLBg1aLPWb9+cMklsNtu6UFDf/tb3hGZmVlXKFIisDwigvRv/q1ycmQAABMPSURBVJyIOJsVKw/WJGkwcA3whYhouI2+iDg/IsZGxNgRI0Z0KuieZOBAuO462GQTOOggmDUr74jMzGxVFSkReEXS14BPAX+U1B9ot509SauRkoDLI+L32egXJG2UTd8ImN9NMfc4w4fDjTem/gMOgIUL843HzMxWTZESgY8DrwNHR8TzpGv9P6y3gCQBFwKPRkR5NbkpwKSsfxJwXdeH23NtuSVMmQJz56aSgddeyzsiMzPrrMIkAhHxfEScFRG3Z8PPRMT/1RGQdFeVxXYHjgD2lHR/1h1AuhVxH0mzgH2y4ULZbTe47DK4+26YNAnefjvviMzMrDOK9GTB9gysHBERdwCqMf9e3RtOz3foofDDH6Y7CTbdNPWbmVnv4kSgjR+V0wmnngpPPQVnngmbbZaeOWBmZr2HEwFbJRKcfTY88wycdFK6o+DAA/OOyszMGtXn6whIWqPRWbs1kD6sf3+44grYeWc47DCYMSPviMzMrFF9PhEA7gKQdGk78x3RhFj6rEGDUnsEI0ak9gnmzMk7IjMza0QREoHVJU0CdpP04cquNFNEPJRjjH3ChhumZwy8/HJqvnjRorZpra1wxhn5xWZmZtUVIRH4PKnBoHWBAyu6iTnG1Sdtuy2cfnqqMzBhArz+ekoCPvYxGDcu7+jMzKxSn68smN0CeIekGRFxYd7xFMGpp6YnDn7ve/Dud8OLL8JVV6XEwMzMepYilAiUXCrpZElXZ91J2eODrRt897uw337wz3/CeuvB7rvnHZGZmVVTpETgZ8Au2evPgJ1JLRJaN2htTXcPlJKBPfeEN97IOyozM6vU5y8NlBkXEe8uG/6rpAdyi6YPK9UJKF0OOPlk+MlPUjLw17/C6qvnHaGZmZUUqUTgLUlblAYkbQ68lWM8fdb06SvWCTjnnPSwoTvvTAmCSwbMzHqOIpUInAa0SppNenjQpsBn8g2pb5o8eeVx55wD73hHSghKpQUuGTAzy19hEoGImCppK2BrUiLwWES8XpouaZ+IuCW3AAvgxBPTI4lPPNHJgJlZT1GkSwNExOsR8WBEPFCeBGR+kEtQBXPCCfDTn8J11/kygZlZT1CoRKAdbmugSZwMmJn1HE4E2rgZ4iZyMmBm1jM4EbDclCcDH/2okwEzszw4EWgzJ+8AiqiUDEyZ4mTAzCwPhblrQFJ/4IPAGMo+d0Sclb1+uPqS1t1OOCG9nnhiSgZ+9zvfTWBm1iyFSQSA64FlwD+At3OOxSo4GTAzy0eREoFREfGuvIOw2sqTgUMPhauvdjJgZtbdilRH4EZJ++YdhNV3wglw7rlw/fUpGXCdATOz7lWkEoG7gT9I6ge8SXpuQETE2vmGZZWOPz69nnCCSwbMzLpbkUoE/hd4D7BWRKwdEUOcBPRcxx/vkgEzs2YoUiIwC3goIvzgoF6iMhl4vfKh0GZmtsqKdGngOeBWSTcC//eTUrp90Hqm449PDRUdf3zb3QRrrJF3VGZmfUeREoGnsm71rLNe4rjj0quTATOzrleYRCAivpV3DNZ55cnAhAnQ2tqWDLS2wvTpMHlyfvGZmfVWhUkEJLVSpWGhiNgzh3CsE447Dv75Tzj77LZk4G9/S40WXXVV3tGZmfVOhUkEgC+X9Q8EPgIszykW66Qf/zi9nn02bLMNLFmSbi+cMCHfuMzMeqvCJAIRMbNi1J2SpuUSjK2SH/8YZs9OdxOssQbMmQMRqVKhmZl1TGFuH5Q0rKwbLmk/YMN2lvmVpPmSHqpYzy2SZmWvQ7s9eFtBayvcdVd64NDbb8NRR8H++8PcuXlHZmbW+xQmEQBmAjOy7m/AqcDR7SxzEbBfxbivAlMjYitgajZsTdLa2lYn4Kc/hRtvhMGD4dZbYfvt4YILUumAmZk1ps8nApLGSdowIjaLiM2BbwGPZd0j9ZaNiNuAlypGHwxcnPVfDBzSxSFbHdOnpySgVCdgr71gyhQ45RTYZRf43OfgAx+AZ57JN04zs95Cff1Be5LuBfaOiJck7QH8FjgJ2BHYNiIObWf5McANEbFDNrw4ItYtm74oIqpeHpB0DHAMwOjRo3d5+umnu+ATWS1vvw2/+AWcdhr06wdnnpkSA9cdMOudJM2MiLF5x9HX9fkSAaB/RJT+1X8cOD8iromIbwBbducbR8T5ETE2IsaOGDGiO9/KSD/+xx0HDz0E48bBscfCvvumyoRmZlZdIRIBSaW7I/YC/lo2rTN3TbwgaSOA7HX+KsZnXWzMGPjLX+DnP4e774Z3vhPOOy+VGJiZ2YqKkAhcAUyTdB3wGnA7gKQtgZc7sb4pwKSsfxJwXVcEaV1LSiUCDz0Eu+6anki4997w1FN5R2Zm1rP0+UQgIk4HvkS6A+C9Za0P9iPVFahJ0hXAXcDWkuZJOhr4PrCPpFnAPtmw9VCbbgo33wy//CXMmJFKB84916UDZmYlfb6yYE8xduzYmDFjRt5hFNozz8Axx8BNN8H73gcXXghbbJF3VGZWiysLNkefLxEwKxk9Oj134MIL4b774F3vgnPOcemAmRWbEwErFCk9ifDhh+H970/PH3j/++GJJ/KOzMwsH04ErJBGjYIbboCLLoIHH0ylAwcemO42KNfaCmeckUuIZmZN4UTACkuCSZNS6cCee6bEYP/94ZJL0vTS44zHjcs3TjOz7uREwApv5MjUkuEll6TWDCdNSrcaHnroio8zNjPri5wImJFKB444AmbNgne8A6ZOhZdeSk8qPOYYuOwyt19gZn2TEwGzMo89lhKAo4+GQYNg6NBUKnDEEemZBGPGwKc/nVo5fPxxt3RoZr1fZx6xa9YnlTdxPGECHH54Gr76ahgxAm67LXU33QSXXpqW2WAD2GOPtm6HHVKbB2ZmvYUfKNQkfqBQz3fGGaliYHmdgNbW1PTx5Mlt4yJSaUApMZg2DebOTdOGDoX3vrctMdhpJ1httY6t38wSP1CoOZwINIkTgb5tzpy2xOC221JdA0iXF3bbLSUFgwfD6ae3lThUlkCY2YqcCDSHE4EmcSJQLM89B7ff3pYY/OMfafyA7GLcrrumcZdfDh/8YH5xmvVkTgSaw4lAkzgRKLaXXoI77khJwW9+kxIFSPUJdtoJ/uM/2roRI/KN1ayncCLQHK7WZNYEw4bBQQelf/9vvglf+Qqss06qkDh4MPz85/CRj8D668O227bdsvj003lHbmZ9ne8aMGuSyjoBH/hA2/Buu8HMmanE4Pbb07hf/jItt8kmqY5BqcRg223Tcw/MzLqCLw00iS8NWEfuGnjrLXjoobbE4Pbb4fnn07Thw9OdCaXEYKed4KyzfEeC9T2+NNAcTgSaxImArYoIePLJFRODJ59M0wYPhq23hkcfTXclHHss3H2370iw3s+JQHM4EWgSJwLW1Z59ti0pKL8zAVIlxK23hne+E0aPXrkbNqxjlxf8DATLgxOB5nAdAbNeauON4eMfTx3AokWpbYQrr0zNKg8fDvffD1OmwLJlKy671lptScGmm66cKIwaBauv3jb/uHErljCU13cws97NiYBZH3H//amxpG98A847L9UbmDAhXVZYsCA1mlSte+ABeOGFFdclwYYbrpgcHHYYHHJIapXx2mvTbZC+7GDW+/nSQJP40oB1p8o7Ejr61MJly2DevHS7Yq2EobJUAdItkBtskJKGDTao37/GGtXf25cdrBZfGmgOlwiY9QHTp6/4oz9hQhqePr2xRGDgQNhyy9RVE5FKAY46Kt32+Mc/pkRj0KBUmvD88/Dgg+n15Zerr2PddVdMDkqvixalkoYf/AAOOCDVdTjyyK677OBEw6w+lwg0iUsErDfrSInDsmUwf35KCkpJwgsvrNhfel2ypPr79euX6jgMHZoqNg4b1lj/uuu2NfLUmditZ3GJQHO4RMDM2tWREoeBA9vqFbTntdfakoSzzkrr3Guv9A9+0aL0aOZFi9IjmR9+OPXXKnEoGTJk5UShpSU91fE974F77oEvfjE94XH69JQ8DB2aLnNUJhHtcWmD9QUuEWgSlwiY1Vb6l37ccamiY71/68uXw+LFbYlCKVkof63WP38+vP12/TgGD25LDIYObeuvfC31z5oFp50Gl14K++8Pt97adaUNTjJcItAsLhEws1xVFtVPmFD/x3TAgHTZYPjwjr/HZz8L558PZ54J73hHShBKSUXptbz/6afT3RiLF9e+jAGptKH0XIYNNoBTT00lDPW6tddeedzgwW3r6e5bNp1oWIkTATPL1apWdGxPZaKx775twxMnNr6e5cvTZYnKxGHx4vTshqlTYezY9CCnl19O3Zw5KYEoDbdXItGv34oJwkYbwX77wVZbpSdJlipqTp2aLsFU69Zcs/a08q47Ew0nGb2LLw00iS8NmOWju3+UGr2sEQFLl7YlBY12//xnqlw5ZEh6yNOyZaluRXtJRXtWWy2VrixblpKPV19NCcfIkelukMGDU1feXzlcbdqg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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data_transformed=normalize(df_under_5f)\n",
    "plot_elbow(data_transformed,'Elbow Method For Optimal k - only 5 features -with normalization - under sampling')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2268\n",
      "2268\n"
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
       "      <th>failure</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">0</th>\n",
       "      <th>0</th>\n",
       "      <td>1130</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"6\" valign=\"top\">1</th>\n",
       "      <th>0</th>\n",
       "      <td>836</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>82</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>157</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                serial_number\n",
       "failure labels               \n",
       "0       0                1130\n",
       "        5                   4\n",
       "1       0                 836\n",
       "        1                  47\n",
       "        2                   4\n",
       "        3                  82\n",
       "        4                   8\n",
       "        5                 157"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(len(data_transformed))\n",
    "clustered_labels=run_kmeans(data_transformed,6)\n",
    "print(len(clustered_labels))\n",
    "display_result_frame(df_test_under,clustered_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Up Sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random over-sampling:\n",
      "1    37122\n",
      "0    37122\n",
      "Name: failure, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "df_bad_over = df_bad_all.sample(count_good, replace=True)\n",
    "df_test_over = pd.concat([df_good_all, df_bad_over], axis=0)\n",
    "\n",
    "print('Random over-sampling:')\n",
    "print(df_test_over.failure.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_over_5f=df_test_over[smart_attributes]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#without normalization\n",
    "plot_elbow(df_over_5f,'Elbow Method For Optimal k - only 5 features -without normalization oversampled')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
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
       "      <th>failure</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">0</th>\n",
       "      <th>0</th>\n",
       "      <td>37034</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>83</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"4\" valign=\"top\">1</th>\n",
       "      <th>0</th>\n",
       "      <td>36470</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>34</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>586</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>32</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                serial_number\n",
       "failure labels               \n",
       "0       0               37034\n",
       "        1                   5\n",
       "        2                  83\n",
       "1       0               36470\n",
       "        1                  34\n",
       "        2                 586\n",
       "        3                  32"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clustered_labels=run_kmeans(df_over_5f,4,random_state=10)\n",
    "display_result_frame(df_test_over,clustered_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgIAAAEWCAYAAAAU6v/cAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nO3deZwcVbn/8c83CZAQ1pCwJYEQiCjgDGJYREUjyCabigIKBkRRQUBFWVyuXBQXXBD0iqKsimAEWWWLEDYvaBL27QchBBIIJBBCAgRC4Pn9cU7fdCY9Mz1Lpqamv+/Xq15dW1c/VV1d9fSpU6cUEZiZmVlj6ld0AGZmZlYcJwJmZmYNzImAmZlZA3MiYGZm1sCcCJiZmTUwJwJmZmYNrEuJgKRDJd1RNRySNut6WMXrznWRNEPSLt2xrN5A0iuSRq+A5S6zP7WYNip/JwO6+3M7oifjkPRDSS9Iem5Ff1YZtLff9ZXfmaQPS5pVNfyQpA+vgM9ZIb/jntRbjgvdqae+/2rtJgL5x7Uo7zSV7jcrMqh65RNHSPpli/H75fHn17mcWyR9YYUE2f5nny9pcYvte0A3Ln8vSf+R9KqkFyVdJGlEB96/3LaJiNUiYnp3xdgoWvmu+7cy70jgOGCLiFi/i5+7zIGlrKr3u7wtf1h0TD0hIraMiFu6sgz/jsurO77/9tRbIrB33mkq3VdXZFAd9ARwQIuM8HPAYwXF0xmntdi+f+3oAmqdUCTtD/wFOAMYCmwJvAHcIWntrgZtndLyu36rlfk2Bl6MiDk9GVwtfenf1org7VNO/t6WWhF1BPaUND0Xaf5MUj8ASf0kfVfSU5LmSLpQ0pp52gWSjsv9w/O/+SPz8GaS5klSK5/3HPAAsFuefwiwI3BV9UySdpD0v5LmS7qvUtQi6VTgg8BvapR27CLpcUkvSfqfSgxtrUuefkie9qKk73R2Q0p6V87k5+fioX2qpp0v6SxJ10p6FRjX4r0CfgH8MCIuiohFEfEc8AXgFeDreb5DJf1L0q8lvSzpUUk7t7VtVHXZJMfxW0nX5Xn+JWl9Sb/K2+1RSe+piutESU9IWijpYUkf7+S2+aRSadVWnXz/jpIm53WeLGnHqmm3SPpBXpeFkm6UNLTGMj4laWqLccdJuqIzMVUtYxdgIrBh3qbn5/E19+E87TBJj+R4p0v6Uh4/GLiualmvSNpQLf5Ra/niyBmSTpB0P/CqpAH5fZdJmivpSUnHVM2/naQpkhZIel4tSunaWNfDJF1dNTxN0oSq4ZmSts79oXQ8OAL4LHB8Xp+rqxa5taT78/f6V0kDW/ncQyXdIenneT99UtIeVdM3lHSV0rFnmqQvVk07WdKlkv4saQFwaB73tzxuoaQHJL1D0klKx4iZknZt7/tqJdb/u+SRv/vK9/hq3iajJK0t6Zr83byU+0fk99TzO15T6Tg2V+nY9V0tPXa3ua06Si1KJ1T7EvOXVfvY2z/H8YKk6cDHWix7TUnnSJot6Rmly2v9qz7nX5JOlzQPOLlGbK3ux/n7fS7vW7dJ2rJqWkePgzPyvvFwnn5eG/tq9fd/sqQJ+btaqHReGFs17zaS7snT/pZ/A+2XnEVEmx0wA9illWmHAndUDQcwCRgCbET6V/6FPO3zwDRgNLAa8HfgT1XTrs79nyH9y/9r1bQr2/r8/J7K/EcCvwd+CJyfxw0HXgT2JCU/H83Dw/L0WypxtliXa4C18rrMBXavY122IJ1odwJWAX4JLGljG55POlm3HL9S/oxvAysDHwEWAptXve9l4P15nQa2eP878zpsUmPZ/w3cWbUNl5ASg5WAA/Jyh7SzbTariuMF4L3AQOBm4ElSqUz//D1Mqnrvp4ANc8wHAK8CG9Tan1p85qj8uQOAw/K22ay9/beVZQ0BXgIOycs7KA+vU7XOTwDvAAbl4Z/UiGMVYB7wrqpl3wN8so3vel7uprY2X573w8CsquH29uGPAZsCAj4EvAZsU2tZtfa7Gp83A7gXGJm3Qb8c83+R9sfRwHRgtzz/ncAhuX81YIc6v4vRwPy8/A2Ap4Bnqqa9BPRrZb/7YYtlzQD+k/evIcAjwJfbOHa8CXyRtJ9+BXgWUJ5+K/Bb0j69Nen3v3OednJ+73457kF53OukPyQDgAtJv4PvkH5XXwSerPr8ur8vWjkGAz8CbsvLXwf4JLAqsDrwN+CKqnlvoe3f8YXAlfm9o0jH7sPr2Vad+P0tEwu1zyOtHXu/DDxK2i+HkM43AQzI068gHf8HA+vm/eFLLY51R+fvaFCN2Frdj0nH/dVJv/tfAfe2+D115Dg4A3iwaj3+Rd6f2/r+Wbqf7ZmX+2PgrjxtZdLv59i8T3wCWEyN88ty613HlzaDdGKbX9V9sY0vcPeq4SOBm3L/TcCRVdM2zzvXANIPonIw+B3wpcqGAC4AvtHGj/kO0g/xeWBN4C7SybE6ETiBfKKueu8NwPh2fiQfqBqeAJxYx7r8F3BJ1bTB+ctoKxF4vWrbvpDHf5BU2tGvat6LgZOr3ndhG9/bB/I6DKwx7cvA41XbcJkfNenHU/kxtLZtqg/If6iadjTwSNXwu4H5bcR5L7Bvrf2pxXyj8ud+E3gYGNGZg1Be1iHAf2ocAA6tWufvttiPr28RR+XAcxZwau7fknTiWqWVz92GdMAeQPohLwTe38q8H2bZg0Gb+3CN918BHFtrWVXfW3uJwOerhrcHnm6xjJOA83L/baQEc2gnvo+ZedscCJyd9793khK+q9rY72olAgdXDZ8G/K6VzzwUmFY1vGpe/vqkg/NbwOpV03/M0uPJycBtLZZ3MjCxanhv0nGzfx5ePS9/rY5+X9RIBEhJ9AxyIlhjeVsDL1UN30Irv2PSCeUNUn2UyrQvAbe0t606+ftbJhZqn0daO/beTFVyB+zK0sR8vbweg6qmH0Q+AefPebqd2Oraj0lJSgBrVu2PdR8H83dXvR57Ak+09/3n/eyfVdO2ABbl/p2AZ1j2WH4HdSQC9V4a2C8i1qrq/tDGvDOr+p8iZefk16daTBsArBcRT5B+NFuTToDXAM9K2pyULd/aVnARsQj4B/Bd0hf4rxazbAx8KherzZc0n3Si3KCt5ZJOxBWvkTLENtclT/u/bRARr5L+ubXl51XbtlIEvSEwMyLebvE5w6uGq7d1Sy/k11rruEHVdEj/wKLF52xI/Z6v6l9UY7iy3ZD0OUn3Vn0PW5HqL9TrW8D/RESrld+qiudekfTZGrO0/P5g+W3b2nff0gXAZ3LR5SHAhIh4o9aMEXF3RLwYEUsi4lrgIlLWXo8292FJe0i6S6koez7pwNKR7VpL9f61MenyQvXnf5u0zwMcTipBeVTpUstetRbYyndzK+ngt1Puv4X0u2/3t19Dvd/bMvNGxGu5dzXS/jEvIhZWzVvPb6/lfv9CLK0Dsqhq+V36vnIR82+Aj0fE3DxuVUm/z8X6C0gntLXUSmXUFoay9N9kRau/hxbbqmVsn636fq+rZ31a0daxt+U5pmJj0j/h2VX76O9JJQMVbR0zoZX9OF+S+InSZc0FpJMzLPud1X0crBFLR465LbfNQKX6Dhuy/LG8vfUF0smru40EHsr9G5H+bZJfN66abyNSMU1lY90K7A+sHBHPSLqVVKyyNulfY3suJGWL/11j2kzSv6kv1pgGKbPriLbWZTbwrsoESauS/gV21LPASEn9qpKByuWWirbi/n/ALFJR/GlV8fQjFSFWX8ceLklVO9BGLK1j0dFt0ypJGwN/AHYmXZp4S9K9pOLReu0KXC/puYi4rNYMEdHe9cuW3x+kdb6+A3FUPusuSYtJCexnclf326l/3VvdhyWtAlxG+r1cGRFvKtVTqCy71nf4KumfXUWtOxNaHlCejIgxtYKLiMeBg/L+9QngUknr5ES4er5a382tpH/Qm5CKu+eT6gC8j3TCq/mRrYzvDs8CQyStXpUMbET6t9Xlz6/j+2rrvcOAy4GvRsQ9VZOOI5VMbh8RzynVq7iHtveBihdIJZobk0rbYPn1rUtEXERKcNtSz77Xmtmkc0zFRlX9M0klAkMjYklrIba18Nb249y/L7ALKQlYk1T615FjV0st1+PZ1mas02yWP5aPJF3mbNOKqCz4LaWKKyNJ1yoqNeAvBr4uaRNJq5F+8H+t+sJuBb5KymQh/Ss4mlRk1FrN6mq3kq6b/rrGtD8De0vaLWd2A5UqR1Vuo3uedD2yXm2ty6XAXpI+IGll4BQ6t53/TfrBHC9pJaWKYXsDl9Tz5rwjfBP4rqTPSBokaX3gj8AawOlVs68LHJM/51OkRObaPK2j26Ytg0k/xMq/mMNIJQId8RCwO/A/qqo82UHXAu/I22WA0u2aW5BKojrjQtIJa0lE1GwHAdJdHJJWU6psuitwMC0qtbahrX14ZdJ1y7nAEqWKXLtWvfd5YB1VVWglJdd7ShqS94uvtfP5/wEWKFUgHJRj2ErStnndDpY0LCet8/N76vndQvrtjiMV6c4Cbid9x+uQTma1dOd+uYyImAn8L/DjvJ2bSP8U2zvB1au976um/K/vMuCiWP7OotVJ/zrnK1WY/n6L6a1ur3x8nQCcKmn1nLB/g7TPrQj3Ap/IpRibkbZtvSaQjlUjlO58OrEyISJmAzcCv5C0Rv6dbSrpQ/UuvI39eHVSkvEiKYn5UQdibs1ReT2GkErXOny3WAt3kmL9aj6u7QtsV88b6z1BXa1l732+vI15ryRVKrqXVFx/Th5/LvAn0on+SdJ18aOr3ncraWNXEoE7SBv8NuoQyU0RMa/GtJmkbO7bpB/fTFIRc2X9zwD2V6q9eWYdH9fqukTEQ8BRpNv2ZpOyxg7fwx0Ri4F9gD1IGftvgc9FxKMdWMZfScXVX8/LeJhUn+L9EVF9ueLfwJg8z6nA/lXTO7pt2ornYdKdDHeSDkzvJlWS6ehy7gP2Av6gTtRezuu2F+lf1IvA8cBeEfFCm29s3Z9ICc2f2pnvWNK/rPnAz0h1bW6pM+ZW9+H8r/UY0kHyJVKpxFVV732UlLxOz0WmG+ZY7yP9u7mRdg5C+WSxN+ny3ZOkfeWPpH9GkE7cD0l6hbTPHBgRr9e5bo+RLg3enocXkCoi/quNPwHnAFvk9enSXRqtOIhUH+RZ0j/w70fExO5YcHvfVxtGkEqevtbieLwRqfLaINL3chfLl2619zs+mvTHYzrp2PsX0nFuRTidVG/qedKltY4kWH8g1Y25D7ibVFG72udIidbDpG17Ke1fAq7W2n58Ibkia172XR1YZmv+QvrtTc9dl9rFyOeMT5ASq/mkPxrXkBKYNlVqyFqDknQoqeLOB4qOpawkDQLmkGp9P150PGbWu0maQTru/nMFf86/SRVmz2trPj9rwKzrvgJMdhJgZkWS9CGl9gsGSBoPNFFH3Se3rGTWBTmzF+mecjOzIm1OuuS0GqmS4P657kSbfGnAzMysgfnSgJmZWQPzpYEeMnTo0Bg1alTRYZiZlcbUqVNfiIhhRcfR1zkR6CGjRo1iypQpRYdhZlYaklq2AGorgC8NmJmZNTAnAmZmZg3MiYCZmVkDcyJgZmbWwJwImJmZNTAnAr3UaafBpEnLjps0KY03MzPrLk4Eeqltt4VPf3ppMjBpUhredtti4zIzs77F7Qj0UuPGwSWXwN57p+6f/4QJE9J4MzOz7tJQJQKSzpU0R9KDNaZ9U1JIGpqHJelMSdMk3S9pm6p5x0t6PHfjV1S8O+8M/fqlhOArX3ESYGZm3a+hEgHgfGD3liMljQQ+CjxdNXoPYEzujgDOyvMOAb4PbA9sB3xf0torIthJk+CNN2DddeGss5avM2BmZtZVDZUIRMRtwLwak04HjgeqH8W4L3BhJHcBa0naANgNmBgR8yLiJWAiNZKLrqrUCdh/f5g3D/7852XrDJiZmXWHhkoEapG0D/BMRNzXYtJwYGbV8Kw8rrXxtZZ9hKQpkqbMnTu3Q3FNnpzqBOyzDyxZAuutl4YnT+7QYszMzNrU0JUFJa0KfAfYtdbkGuOijfHLj4w4GzgbYOzYsTXnac3xx6fXRx5Jr/fdB+PHu56AmZl1r0YvEdgU2AS4T9IMYARwt6T1Sf/0R1bNOwJ4to3xK8SYMTBwINx//4r6BDMza2QNnQhExAMRsW5EjIqIUaST/DYR8RxwFfC5fPfADsDLETEbuAHYVdLauZLgrnncCjFgAGy1VSoRMDMz624NlQhIuhi4E9hc0ixJh7cx+7XAdGAa8AfgSICImAf8AJicu1PyuBWmuTklAtGhiwtmZmbta6g6AhFxUDvTR1X1B3BUK/OdC5zbrcG1oakJzjkHZs+GDTfsqU81M7NG0FAlAmXV3JxeXU/AzMy6mxOBEmhqSq+uJ2BmZt3NiUAJrL02jBzpRMDMzLqfE4GSqFQYNDMz605OBEqiuRn+3/+D118vOhIzM+tLnAiURHMzvPUWPPxw0ZGYmVlf4kSgJFxh0MzMVgQnAiWx2WYwaJATATMz615OBEqif39497vdloCZmXUvJwIl4qaGzcysuzkRKJGmJpg3D555puhIzMysr3AiUCKVpoZdT8DMzLqLE4ESqdw54HoCZmbWXZwIlMiaa8KoUS4RMDOz7uNEoGSampwImJlZ93EiUDLNzfDYY7BoUdGRmJlZX+BEoGSam+Htt+Ghh4qOxMzM+gInAiXjOwfMzKw7OREomdGjYfBgJwJmZtY9GiYRkHSupDmSHqwa9zNJj0q6X9LlktaqmnaSpGmS/p+k3arG757HTZN0Yk+vR79+qalhJwJmZtYdGiYRAM4Hdm8xbiKwVUQ0AY8BJwFI2gI4ENgyv+e3kvpL6g/8D7AHsAVwUJ63RzU3p7YE3NSwmZl1VcMkAhFxGzCvxbgbI2JJHrwLGJH79wUuiYg3IuJJYBqwXe6mRcT0iFgMXJLn7VHNzTB/Psyc2dOfbGZmfU3DJAJ1+DxwXe4fDlSfZmflca2Nr0nSEZKmSJoyd+7cbgu00sKgLw+YmVlXOREAJH0HWAJcVBlVY7ZoY3xNEXF2RIyNiLHDhg3reqCZEwEzM+suA4oOoGiSxgN7ATtH/N9V91nAyKrZRgDP5v7WxveY1VdPdw/4mQNmZtZVDV0iIGl34ARgn4h4rWrSVcCBklaRtAkwBvgPMBkYI2kTSSuTKhRe1dNxQ6on4BIBMzPrqlImApKOlbSGknMk3S1p13beczFwJ7C5pFmSDgd+A6wOTJR0r6TfAUTEQ8AE4GHgeuCoiHgrVyz8KnAD8AgwIc/b45qa4PHH4dVXi/h0MzPrK8p6aeDzEXFGvr9/GHAYcB5wY2tviIiDaow+p435TwVOrTH+WuDaDkfczZqb0+2DDz4I229fdDRmZlZWpSwRYGmlvT2B8yLiPmpX5OuzKk0Nu56AmZl1RVkTgamSbiQlAjdIWh14u+CYetSoUanSoOsJmJlZV5T10sDhwNbA9Ih4TdI6pMsDDcNNDZuZWXcoa4lAkJr4PSYPDwYGFhdOMdzUsJmZdVVZE4HfAu8DKhUAF5KeAdBQmpthwQJ46qmiIzEzs7IqayKwfUQcBbwOEBEvASsXG1LPq1QY9OUBMzPrrLImAm/mJwEGgKRhNFhlQYCttgLJiYCZmXVeWROBM4HLgXUlnQrcAfyo2JB63mqrwaab+hZCMzPrvFLeNRARF0maCuxMaj9gv4h4pOCwCuGmhs3MrCtKWSIgaQfgmYj4n4j4DTBLUkO2r9fcDE88Aa+8UnQkZmZWRqVMBICzgOpT36t5XMNpakq3Dz7wQNGRmJlZGZU1EVDVI4OJiLcp6WWOrnJTw2Zm1hVlTQSmSzpG0kq5OxaYXnRQRdh4Y1hzTdcTMDOzzilrIvBlYEfgGWAWsD1wRKERFURKlwecCJiZWWeUsjg9IuYABxYdR2/R1AQXXABvv52eQWBmZlavUiYCuQGhLwKjqFqHiPh8UTEVqbk53TUwYwaMHl10NGZmVialTASAK4HbgX8CbxUcS+Gqmxp2ImBmZh1R1kRg1Yg4oeggeostt1za1PDHP150NGZmViZlvaJ8jaQ9iw6itxg8GMaMcYVBMzPruLImAseSkoFFkhZIWihpQXtvknSupDmSHqwaN0TSREmP59e183hJOlPSNEn3S9qm6j3j8/yPSxq/Qtawg5qb3ZaAmZl1XCkTgYhYPSL6RcSgiFgjD69Rx1vPB3ZvMe5E4KaIGAPclIcB9gDG5O4IcsuFkoYA3yfdsrgd8P1K8lCk5maYPh0WtJsOmZmZLVXKRABA0tqStpO0U6Vr7z0RcRswr8XofYELcv8FwH5V4y+M5C5gLUkbALsBEyNiXkS8BExk+eSixzU1pVc3NWxmZh1RykRA0heA24AbgP/Oryd3cnHrRcRsgPy6bh4/HJhZNd+sPK618bXiPELSFElT5s6d28nw6lN954CZmVm9SpkIkOoIbAs8FRHjgPcA3X2mVY1x0cb45UdGnB0RYyNi7LBhw7o1uJZGjoS11nI9ATMz65iyJgKvR8TrAJJWiYhHgc07uaznc5E/+XVOHj8LGFk13wjg2TbGF0pKpQIuETAzs44oayIwS9JawBXARElX0vmT8VVApeb/eFJjRZXxn8t3D+wAvJwvHdwA7JrrKKwN7JrHFa6pKdURePvtoiMxM7OyKGWDQhFRaTbnZEmTgDWB69p7n6SLgQ8DQyXNItX+/wkwQdLhwNPAp/Ls1wJ7AtOA14DD8mfPk/QDYHKe75SIaFkBsRDNzfDqq/DEE6ldATMzs/aUMhGQ9KeIOAQgIm6tjAMOaet9EXFQK5N2rjFvAEe1spxzgXM7EnNPqFQYvP9+JwJmZlafsl4a2LJ6QFJ/4L0FxdJrbLllevqg6wmYmVm9SpUISDpJ0kKgKbcouCAPz2Hptf2GNWgQvOMdTgTMzKx+pUoEIuLHEbE68LPcomClVcF1IuKkouPrDXzngJmZdUSpEoEq10gaDCDpYEm/lLRx0UH1Bs3N8NRT8PLLRUdiZmZlUNZE4CzgNUnNwPHAU8CFxYbUO1RXGDQzM2tPWROBJblW/77AGRFxBrB6wTH1CpVnDvjygJmZ1aOUtw8CCyWdBBwM7JTvGlip4Jh6heHDYcgQJwJmZlafspYIHAC8ARweEc+RHvrzs2JD6h0qTQ370oCZmdWjlIlARDwXEb+MiNvz8NMR4ToCWXNzamr4rbeKjsTMzHq7UiUCku7Irwur2hFYUBkuOr7eoqkJFi2CadOKjsTMzHq7UtURiIgP5FdXDGxD5c6B++6DzTv7TEYzM2sIZSsRGNJWV3R8vcUWW0D//q4nYGZm7StViQAwFQhAwEbAS7l/LdKTAzcpLrTeY+BAeOc7feeAmZm1r1QlAhGxSUSMBm4A9o6IoRGxDrAX8Pdio+tdmpqcCJiZWftKlQhU2TYirq0MRMR1wIcKjKfXaW6GmTNh3ryiIzEzs96srInAC5K+K2mUpI0lfQd4seigepNKhcEHHig2DjMz693KmggcBAwDLs/dsDzOsuo7B8zMzFpTtsqCAETEPODY1qZL+nVEHN2DIfU6668PQ4c6ETAzs7aVtUSgPe8vOoCiVZoadiJgZmZt6auJgJESgYcegiVLio7EzMx6KycCgKSvS3pI0oOSLpY0UNImkv4t6XFJf5W0cp53lTw8LU8fVWz0rWtuhtdfh8cfLzoSMzPrrfpqIqC6Z5SGA8cAYyNiK6A/cCDwU+D0iBhDarjo8PyWw4GXImIz4PQ8X6/U1JRefXnAzMxa01cTgTM6OP8AYJCkAcCqwGzgI8ClefoFwH65f988TJ6+s6S6E4+e9K53wYABTgTMzKx1pbprQNLVpCaGa4qIffLr+fUuMyKekfRzUhPFi4AbSU0Zz4+IytX1WcDw3D8cmJnfu0TSy8A6wAs14j0COAJgo402qjekbrPKKikZ8DMHzMysNWUrEfg58AvgSdJJ+w+5ewV4sDMLlLQ26V/+JsCGwGBgjxqzVhKQWv/+ayYnEXF2RIyNiLHDhg3rTHhd5jsHzMysLaVKBCLi1oi4FXhPRBwQEVfn7jPABzq52F2AJyNibkS8SXpmwY7AWvlSAcAI4NncPwsYCZCnrwn02oZ8m5rgmWfgRbe7aGZmNZQqEagyTNLoyoCkTUitC3bG08AOklbN1/p3Bh4GJgH753nGA1fm/qvyMHn6zRHR6uWKormFQTMza0up6ghU+Tpwi6TpeXgU8KXOLCgi/i3pUuBuYAlwD3A28A/gEkk/zOPOyW85B/iTpGmkkoADO7sSPaGSCNx/P3zkI8XGYmZmvU8pE4GIuF7SGOCdedSjEfFGF5b3feD7LUZPB7arMe/rwKc6+1k9bb31UucSATMzq6WUlwYkrQp8C/hqRNwHbCRpr4LD6rWampwImJlZbaVMBIDzgMXA+/LwLOCHxYXTu1WaGn7zzaIjMTOz3qasicCmEXEa8CZARCyiA60JNprmZli8GB57rOhIzMystylrIrBY0iDy/fuSNgU6XUegr3NTw2Zm1pqyJgLfB64HRkq6CLgJOL7YkHqvd74TVlrJiYCZmS2vdHcN5Hv9HwU+AexAuiRwbEQs18SvJSuvDFts4UTAzMyWV7pEICJC0hUR8V7Svf5Wh+ZmmDix6CjMzKy3KeulgbskbVt0EGXS1ASzZ8PcuUVHYmZmvUlZE4FxwJ2SnpB0v6QHJPkZe21wU8NmZlZL6S4NZLWeDmhtqE4Edtml2FjMzKz3KGUiEBFPAUhaFxhYcDilMGwYbLBBeuaAmZlZRSkvDUjaR9LjwJPArcAM4LpCgyoBNzVsZmYtlTIRAH5AunXwsYjYhPTo4H8VG1Lv19wMDz+cWhk0MzOD8iYCb0bEi0A/Sf0iYhKwddFB9XbNzel5A48+WnQkZmbWW5Q1EZgvaTXgNuAiSWcASwqOqderVBh0PQEzM6soayKwL7AI+DqpqeEngL0LjagE3vGO1Mqg6wmYmVlFWe8aeLVq8ILCAimZlVaCLbd0ImBmZkuVskRA0kJJC3L3uqS3JC0oOq4yaG52ImBmZkuVMhGIiNUjYo3cDQQ+Cfym6LjKoLkZ5syB558vOhIzM+sNSpkItMAnQScAABfQSURBVBQRVwAf6ez7Ja0l6VJJj0p6RNL7JA2RNFHS4/l17TyvJJ0paVpu3nibbluRHtDUlF5dKmBmZlDSREDSJ6q6/SX9BIguLPIM4PqIeCfQDDwCnAjcFBFjgJvyMKTmjcfk7gjgrC58bo/zMwfMzKxaKSsLsuwdAktILQvu25kFSVoD2Ak4FCAiFgOLJe0LfDjPdgFwC3BC/pwLIyJIT0FcS9IGETG7M5/f09ZZB4YPdyJgZmZJKROBiDisGxc3GpgLnCepGZgKHAusVzm5R8Ts/FwDgOHAzKr3z8rjlksEJB1BKjVgo4026saQu6a52W0JmJlZUspEQNKZbU2PiGM6sLgBwDbA0RHx79w40YltzK9aH9lKHGcDZwOMHTu2K5cuulVTE9x4I7zxBqyyStHRmJlZkUpZR4D0xMFtgMdztzXwFunf/NQOLmsWMCsi/p2HL83Lfl7SBgD5dU7V/COr3j8CeLYT61CY5mZYsgQeeaToSMzMrGhlTQTGAOMi4tcR8WvSQ4e2jogLIqJDDQxFxHPATEmb51E7Aw8DVwHj87jxwJW5/yrgc/nugR2Al8tSP6DCFQbNzKyilJcGgA2B1YF5eXi1PK6zjiY9s2BlYDpwGClJmiDpcOBp4FN53muBPYFpwGt53lIZMwYGDnQ9ATMzK28i8BPgHkmT8vCHgJM7u7CIuBcYW2PSzjXmDeCozn5WbzBggJsaNjOzpJSJQEScJ+k6YPs86sRcxG91am6Gq66CCFCt6o9mZtYQSllHQNL7gYURcSXpEsHxkjYuOKxSaW6GF16A55w+mZk1tFImAqTW/F7L9/1/C3gKuLDYkMrFFQbNzAzKmwgsydfq9wXOjIgzSCUDVofTToOXX079lURg0qQ03szMGktZE4GFkk4CDgb+Iak/sFLBMZXGttvC4YfDsGEpEZg0CT796TTezMwaS1kTgQOAN4DDcyXB4cDPig2pPMaNgwkTUqnAP/+ZkoAJE9J4MzNrLKVMBCLiuYj4ZUTcnoefjoj/qyMg6c7ioiuHcePgfe+DuXPhox91EmBm1qhKmQjUYWDRAfR2kybBgw/CeuvBJZfAH/9YdERmZlaEvpoI9JoH/PRGlToBf/sbTJ2aHk38pS+lYTMzayx9NRGwNkyevLROwPDhMHEirLwyfO1r8MorRUdnZmY9qVSJgKR6H5rrtvLacPzxy9YJ2HpruOyy1LjQZz4Db71VXGxmZtazSpUIAHcCSPpTO/Md0gOx9Cl77gm//jVcfTV84xtFR2NmZj2lbM8aWFnSeGBHSZ9oOTEi/p5fH+zxyPqAI4+EadPg9NNh003hmGOKjsjMzFa0siUCXwY+C6wF7N1iWgB/7/GI+pif/QymT4evfx022QT2brmVzcysTylVIhARdwB3SJoSEecUHU9f1L8/XHQRfPjDcOCBcPvtsM02RUdlZmYrStnqCFT8SdIxki7N3dGS3MRwNxk8ONUVGDoU9toLZs4sOiIzM1tRypoI/BZ4b379LbAN6YmE1k3WXx/+8Q949VX42MdgwYKiIzIzsxWhVJcGqmwbEc1VwzdL8gN1u9lWW6VGhvbcEw44IJUSDCjrHmNmZjWVtUTgLUmbVgYkjQZ89/sKsOuucNZZcP31cPTREG6z0cysTynr/7tvAZMkTSc1HrQxcFhXFpgfZTwFeCYi9pK0CXAJMAS4GzgkIhbnRo0uJF2aeBE4ICJmdOWze7svfhGeeAJ++lPYbDM47riiIzIzs+5SyhKBiLgJGAMck7vNI2JSZbqkj3ZisccCj1QN/xQ4PSLGAC8Bh+fxhwMvRcRmwOl5vj7vRz+C/feHb30L/u6bNM3M+oxSJgIAEfFGRNwfEfdFxBstJnfo5CxpBPAx4I95WMBHgEvzLBcA++X+ffMwefrOef4+rV8/uPBC2G47OPhg+M9/io7IzMy6Q2kTgXZ09MT8K+B44O08vA4wPyKW5OFZwPDcPxyYCZCnv5zn7/MGDYKrrkp3FOy9N8yYUXREZmbWVX01Eai7SpukvYA5ETG1enQby2xrWstlHyFpiqQpc+fOrTekXm3dddNthYsXp9sK588vOiIzM+uKvpoIdMT7gX0kzSBVDvwIqYRgLUmVypQjgGdz/yxgJECeviYwr9aCI+LsiBgbEWOHDRu24tagh73rXamewGOPpXoDb75ZdERmZtZZfTURmFHvjBFxUkSMiIhRwIHAzRHxWWASsH+ebTxwZe6/Kg+Tp98c0Xg31Y0bB3/4A9x0E3zlK76t0MysrEp5+2C+1e9jwCiq1iEifplfl3syYSecAFwi6YfAPUDl2QbnkJo4nkYqCTiwGz6rlA49NN1W+MMfpqcVnnRS0RGZmVlHlTIRAK4GXgceYGkFvy6LiFuAW3L/dGC7GvO8Dnyquz6z7E45JT2t8NvfhtGjUwuEZmZWHmVNBEZERFPRQRhIcM458PTTMH48jBwJO+5YdFRmZlavstYRuE7SrkUHYcnAgXD55SkJ2HffdLnAzMzKoayJwF3A5ZIWSVogaaEkPx+vQEOHwrXXwttvp9sK59W8j8LMzHqbsiYCvwDeB6waEWtExOoRsUbRQTW6MWPgiitg2rR0V8EbVe09TpoEp51WXGxmZlZbWROBx4EHG/G2vd7ugx+EE06A++9PrQ9GpCTg05+GbbctOjozM2uprJUFZwO3SLoO+L//nZXbB61Yp54Kzz4L558P73kPPPMMTJiQSgnMzKx3KWsi8GTuVs6d9TLnnpsqDd5+e3o2wSabFB2RmZnVUspEICL+u+gYrG233AKPPJLaFZgwAZqa4NJLYVff62Fm1quUso6ApEmSbm7ZFR2XJZU6ARMmwCWXwAUXwGuvwW67wY9+lO4sMDOz3qGUJQLAN6v6BwKfBJa0Mq/1sMmTl60TcMghsM46qfXB73wH/vOflBysuWaxcZqZGaivVLyXdGtEfKjoOFozduzYmDJlStFhFCoCzjwTjjsuPZvg8sthiy2KjsrMeitJUyNibNFx9HVlvTQwpKobKml3YP2i47K2SXDssXDzzfDyy7DddvC3vxUdlZlZYytlIgBMBabk7n+BbwCHFxqR1W2nnWDq1FSB8NOfhm9+E5b4wo6ZWSFKlQhI2lbS+hGxSUSMBv4beDR3DxcbnXXE8OHpzoKjjoJf/AI++lGYM6foqMzMGk+pEgHg98BiAEk7AT8GLgBeBs4uMC7rhJVXht/8JlUcvOsu2Gab9GpmZj2nbIlA/4ioPM7mAODsiLgsIr4HbFZgXNYFn/sc3HlnSgx22gl+97tUsdDMzFa80iUCkiq3PO4MVLcdUNZbIQ3YemuYMgV23hm+8hU4/HBYtKjoqMzM+r6yJQIXA7dKuhJYBNwOIGkz0uUBK7EhQ+Caa+B734PzzoMPfABmzCg6KjOzvq1UiUBEnAocB5wPfKDq6YP9gKOLisu6T//+cMopcNVV6VkF730v3Hhj0VGZmfVdpUoEACLiroi4PCJerRr3WETcXWRc1r323jtdKthwQ9h9dzdNbGa2opQuEehukkbmZxc8IukhScfm8UMkTZT0eH5dO4+XpDMlTZN0v6Rtil2DvmuzzdJdBAcemJom/sQnUkNEZmbWfRo+ESA9o+C4iHgXsANwlKQtgBOBmyJiDHBTHgbYAxiTuyOAs3o+5MYxeDBcdBH86lfwj3+k5OC885adZ9IkOO20YuIzMyu7hk8EImJ25bJCRCwEHgGGA/uS2iggv+6X+/cFLozkLmAtSRv0cNgNpbpp4rfegs9/Hv7rv9K0ypMOt9222BjNzMqq4ROBapJGAe8B/g2sFxGzISULwLp5tuHAzKq3zcrjai3vCElTJE2ZO3fuigq7YXzwg/Dgg7DllvCDH6QmivfbLzVIVHnSoZmZdYwTgUzSasBlwNciYkFbs9YYV7P5m4g4OyLGRsTYYcOGdUeYDW/DDeHuu2H77eGBB2DBgpQM7Lwz/Pzn8NBDbozIzKwjnAgAklYiJQEXRcTf8+jnK0X++bXSEv4sYGTV20cAz/ZUrAb/+le6tfDb34Y110yVCOfMgW99C7baCjbeGL70JbjiCli4sOhozcx6t4ZPBCQJOAd4JCJ+WTXpKmB87h8PXFk1/nP57oEdgJcrlxBsxavUCZgwAU49FS6/HG66Cc48E55+Gs4+G8aOhYsvho9/PDVSNG5cqkz4wAMuLTAza0nR4EdGSR8gtVD4AFC5U/3bpHoCE4CNgKeBT0XEvJw4/AbYHXgNOCwiprT3OWPHjo0pU9qdzdpx2mmpYmB1nYBJk2DyZDj++KXjFi+G//1fuO661D3wQBo/YkRql2CPPWCXXWCNNXo2fjOrn6SpETG26Dj6uoZPBHqKE4FiPfMMXH99SgomTkx1CwYMgB13TEnBHnuk6dtt136SYWY9w4lAz3Ai0EOcCPQeb76ZnnZYKS247740fp114NVX4YQT4Gtfg3vuWXoZwnclmPU8JwI9w4lAD3Ei0Hs9++zS0oLrrkvJgJSee3DwwfDlL6d6B/37Fx2pWWNxItAznAj0ECcC5bBkCXzhC6ltgg02gOeeSxUMhwxJdQp23RV22y3VNTCzFcuJQM8YUHQAZr3J7benpoy/9z046yy47DJ4/fX0BMQbbkiXCQC22GJpUrDTTrDqqsXGbWbWWU4EzLLqWxPHjUtdZfi881LJwIMPLk0KzjorPQNhlVVSq4e77ZaSg3e/O11aMDMrg4ZvR8CsYvLkZSsGjhuXhidPTsNSOskfd1xKBl56KdUtOPJImD07NWjU3JxaPxw/Hv7yF6huWfq001KyUc0PTDKzormOQA9xHYG+b9asdGviDTek13nz0vhttkmlBeuumxpBqiQbLUsgzGxZriPQM5wI9BAnAo3lrbfSMxFuuCF1d96Zxg0cCG+/ndovmDoVfvKTlAyss44vJ5i15ESgZzgR6CFOBBrbggXpMco33giXXJIuK1RbYw0YPTp1m26aukr/Rhulxo/aUm+Li2Zl4kSgZ7iyoFkPWGON9JTENdeEv/0NTjwRfv/7VN9g8OD0EKXp09PTE6+5JjWRXNG/f3qQUq0kYfTotOxtt132MkP1ZQczs7Y4ETDrIS3rBOy669Lhr31t6Xxvv52aRJ4+PSUIlSThiSfg0kvhxReXXe7QoSkpaGqCvfaCj3wEbrsNfvzjlEC88kpKNrpy6cElDmZ9lxMBsx7S1l0J1SfYfv1g5MjUfehDyy/n5ZdrJwnTp8Nrr6USBYCjjlr6noEDU2XFddeFYcPaf23ZLoJLHMz6LtcR6CGuI2ArWuXkfNBB8Kc/pX/qG2wAc+ak2xhrvb7+eu1lDR68fHKwaBFcfXVqYXHSpFTRcffdU4nEaqt1vsTBpQ3WGtcR6BkuETDrA1pedvj4x5cOH3po7fdEpOcqtJUozJ2bLlPce28aXrwYrrwyvf/II5cua5VVUkJQ6YYNa7t/nXVg5ZXTe1d0aYMTDbO2OREw6wPqvexQTUr/5FdbLVU6bM/NNy8tcfjzn1MDShtuCC+8kBKGF15Y2j91anqdP7/15a2xxtLEYNNN06Ogt94a7r8/fca998Jjj6XLFIMGpddK13J41VVhpZVqf86KTDScZFhf4EsDPcSXBqzMWpY41NsY0ptvpoaVWiYKtfqnTYOFCzsfY//+yycHlYRh0aL0uOl3vCN9zp57pv7WkovW+lddNZV+VC6DdHa71GNFJxllSGJ8aaBnOBHoIU4ErMxW9EmjcgL9ylfSMxwuugi22y5VfnzttXQir/TXM1xr3IwZKfEYPDglDYsWpUSlo6SlycGgQWnc7Nmw/vrw/POpJckRI1IFzUGDln2td9ygQalBqqOOggsvTHeY3H5797ZEWYYkxolAz3Ai0EOcCJjVtiJPSC0/o5JoVJb95pspIahOGtpKMlrrv+ceePzxlACst16a9vrrS18r/V01cGBKZCoJRHW3yiq1x7c137RpcMYZ8LGPwbXXwsknw/bbp8ss9Xb9ajyxpru+UycCPcOJQA9xImBWW0+VNqyoRKO1JKOliFTZsmVyUOu1uv+KK+Cmm+D9708n6TfeWDpPW13L+ZYs6fq61tKvX+0EYcmSdMlnvfXSendmezsR6BlOBDpJ0u7AGUB/4I8R8ZO25nciYFaMFZlo9JYkox5LliyfHNx2G3zjG+kuk7//Hb73Pdhyy1RS0h3d3Xen1jK/9z045ZSOx+xEoIdEhLsOdqST/xPAaGBl4D5gi7be8973vjfMrG/56U8jbr552XE335zGd9XNN0cMHbp0+S2Hy7L8732v88sFpkQvOOb39a7G1R2rw3bAtIiYHhGLgUuAfQuOycx62PHHL/8Pfdy47rmk0dYtod1hRS6/umTklFPS66c/ncZb7+NLA50gaX9g94j4Qh4+BNg+Ir7aYr4jgCMANtpoo/c+9dRTPR6rmVlP810D5eIGhTqnVmOqy2VUEXE2cDakOgIrOigzs96g1sl+3LjuuwvEupcvDXTOLGBk1fAI4NmCYjEzM+s0JwKdMxkYI2kTSSsDBwJXFRyTmZlZh/nSQCdExBJJXwVuIN1BcG5EPFRwWGZmZh3mRKCTIuJa4Nqi4zAzM+sKXxowMzNrYL59sIdImgv0xvsHhwIvFB1EJzn2Yjj2nlfWuKFrsW8cEcO6MxhbnhOBBidpSlnv03XsxXDsPa+scUO5Y28UvjRgZmbWwJwImJmZNTAnAnZ20QF0gWMvhmPveWWNG8ode0NwHQEzM7MG5hIBMzOzBuZEwMzMrIE5EWhQkkZKmiTpEUkPSTq26Jg6QlJ/SfdIuqboWDpC0lqSLpX0aN727ys6pnpJ+nreVx6UdLGkgUXH1BpJ50qaI+nBqnFDJE2U9Hh+XbvIGFvTSuw/y/vM/ZIul7RWkTG2plbsVdO+KSkkDS0iNmudE4HGtQQ4LiLeBewAHCVpi4Jj6ohjgUeKDqITzgCuj4h3As2UZB0kDQeOAcZGxFakZ2wcWGxUbTof2L3FuBOBmyJiDHBTHu6Nzmf52CcCW0VEE/AYcFJPB1Wn81k+diSNBD4KPN3TAVn7nAg0qIiYHRF35/6FpBPS8GKjqo+kEcDHgD8WHUtHSFoD2Ak4ByAiFkfE/GKj6pABwCBJA4BV6cWP3o6I24B5LUbvC1yQ+y8A9uvRoOpUK/aIuDEiluTBu0iPPu91WtnuAKcDxwOund4LOREwJI0C3gP8u9hI6vYr0kHl7aID6aDRwFzgvHxZ44+SBhcdVD0i4hng56R/dLOBlyPixmKj6rD1ImI2pEQYWLfgeDrr88B1RQdRL0n7AM9ExH1Fx2K1ORFocJJWAy4DvhYRC4qOpz2S9gLmRMTUomPphAHANsBZEfEe4FV6b/H0MvL19H2BTYANgcGSDi42qsYj6Tuky3oXFR1LPSStCnwH+K+iY7HWORFoYJJWIiUBF0XE34uOp07vB/aRNAO4BPiIpD8XG1LdZgGzIqJS8nIpKTEog12AJyNibkS8Cfwd2LHgmDrqeUkbAOTXOQXH0yGSxgN7AZ+N8jQAsykpebwv/2ZHAHdLWr/QqGwZTgQalCSRrlU/EhG/LDqeekXESRExIiJGkSqr3RwRpfhnGhHPATMlbZ5H7Qw8XGBIHfE0sIOkVfO+szMlqehY5SpgfO4fD1xZYCwdIml34ARgn4h4reh46hURD0TEuhExKv9mZwHb5N+C9RJOBBrX+4FDSP+o783dnkUH1QCOBi6SdD+wNfCjguOpSy7FuBS4G3iAdOzotU3HSroYuBPYXNIsSYcDPwE+KulxUg32nxQZY2taif03wOrAxPxb/V2hQbailditl3MTw2ZmZg3MJQJmZmYNzImAmZlZA3MiYGZm1sCcCJiZmTUwJwJmZmYNzImAWYOSNKrWU+LMrLE4ETAzM2tgTgTMDEmj84OQti06FjPrWU4EzBpcbvL4MuCwiJhcdDxm1rMGFB2AmRVqGKnN/U9GxENFB2NmPc8lAmaN7WVgJunZE2bWgFwiYNbYFgP7ATdIeiUi/lJ0QGbWs5wImDW4iHhV0l6kJ9u9GhGleTyvmXWdnz5oZmbWwFxHwMzMrIE5ETAzM2tgTgTMzMwamBMBMzOzBuZEwMzMrIE5ETAzM2tgTgTMzMwa2P8HFKcO8c0BctEAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data_transformed=normalize(df_over_5f)\n",
    "plot_elbow(data_transformed,'Elbow Method For Optimal k - only 5 features -with normalization - under sampling')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
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
       "      <th>failure</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"5\" valign=\"top\">0</th>\n",
       "      <th>0</th>\n",
       "      <td>37003</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"6\" valign=\"top\">1</th>\n",
       "      <th>0</th>\n",
       "      <td>27673</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1566</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5215</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>66</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>132</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2470</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                serial_number\n",
       "failure labels               \n",
       "0       0               37003\n",
       "        1                   9\n",
       "        2                  77\n",
       "        3                   4\n",
       "        5                  29\n",
       "1       0               27673\n",
       "        1                1566\n",
       "        2                5215\n",
       "        3                  66\n",
       "        4                 132\n",
       "        5                2470"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clustered_labels=run_kmeans(data_transformed,6,random_state=10)\n",
    "display_result_frame(df_test_over,clustered_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import PCA\n",
    "# df_allf=d_copy.iloc[:,5:]\n",
    "df_allf_norm=normalize(d_copy.iloc[:,5:])\n",
    "pca = PCA(n_components=2)\n",
    "X = pca.fit_transform(df_allf_norm)\n",
    "\n",
    "labels=d_copy['failure']    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'feature 2')"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(X[:,0],X[:,1],c=labels,marker='x')\n",
    "plt.title(\"First 2 features after PCA. Taking all features. Normalized. Last day alone\")\n",
    "plt.xlabel(\"feature 1\")\n",
    "plt.xlabel(\"feature 2\")\n",
    "# X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(38256, 5)\n",
      "(38256, 5)\n",
      "38256\n"
     ]
    }
   ],
   "source": [
    "# df_allf=d_copy.iloc[:,5:]\n",
    "df_5f_norm=normalize(d_5f)\n",
    "print(d_5f.shape)\n",
    "print(df_5f_norm.shape)\n",
    "pca = PCA(n_components=2)\n",
    "X = pca.fit_transform(df_5f_norm)\n",
    "labels=d_copy['failure']   \n",
    "print(len(d_copy))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'feature 2')"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(X[:,0],X[:,1],c=labels,marker='x')\n",
    "plt.title(\"First 2 features after PCA. Taking 5 features. Normalized.Last day alone\")\n",
    "plt.xlabel(\"feature 1\")\n",
    "plt.xlabel(\"feature 2\")\n",
    "# X.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Trying for 10 days"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(382305, 24)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_csv(\"../dataset/ST12000NM0007_last_10_day_all_q_raw.csv\")\n",
    "dataset.shape"
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
      "Good entry count:  371048 \n",
      "Failed entry count:  11257\n"
     ]
    }
   ],
   "source": [
    "d_copy=dataset.copy()\n",
    "count_good, count_bad = dataset.failure.value_counts()\n",
    "print(\"Good entry count: \",count_good,\"\\nFailed entry count: \", count_bad)\n",
    "\n",
    "# Divide by class\n",
    "df_good_all = d_copy[d_copy['failure'] == 0]\n",
    "df_bad_all = d_copy[d_copy['failure'] == 1]\n"
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
      "Random under-sampling:\n",
      "1    11257\n",
      "0    11257\n",
      "Name: failure, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "df_good_under = df_good_all.sample(count_bad)\n",
    "df_test_under = pd.concat([df_good_under, df_bad_all], axis=0)\n",
    "\n",
    "print('Random under-sampling:')\n",
    "print(df_test_under.failure.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_under_5f=df_test_under[smart_attributes]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#without normalization\n",
    "plot_elbow(df_under_5f,'Elbow Method For Optimal k - only 5 features -without normalization undersampleed last 10 days')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
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
       "      <th>failure</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">0</th>\n",
       "      <th>0</th>\n",
       "      <td>11256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">1</th>\n",
       "      <th>0</th>\n",
       "      <td>11237</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                serial_number\n",
       "failure labels               \n",
       "0       0               11256\n",
       "        1                   1\n",
       "1       0               11237\n",
       "        1                  20"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clustered_labels=run_kmeans(df_under_5f,2,random_state=10)\n",
    "display_result_frame(df_test_under,clustered_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random over-sampling:\n",
      "1    371048\n",
      "0    371048\n",
      "Name: failure, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Increase failed sample count\n",
    "df_bad_over = df_bad_all.sample(count_good, replace=True)\n",
    "df_test_over = pd.concat([df_good_all, df_bad_over], axis=0)\n",
    "\n",
    "print('Random over-sampling:')\n",
    "print(df_test_over.failure.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df_over_5f=df_test_over[smart_attributes]\n",
    "plot_elbow(df_over_5f,'Elbow Method For Optimal k - only 5 features- up sampling')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
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
       "      <th>failure</th>\n",
       "      <th>labels</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">0</th>\n",
       "      <th>0</th>\n",
       "      <td>371008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">1</th>\n",
       "      <th>0</th>\n",
       "      <td>370381</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>667</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                serial_number\n",
       "failure labels               \n",
       "0       0              371008\n",
       "        1                  40\n",
       "1       0              370381\n",
       "        1                 667"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clustered_labels=run_kmeans(df_over_5f,2,random_state=10)\n",
    "display_result_frame(df_test_over,clustered_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Trying to cluster by Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_all_models(number_of_days):\n",
    "    st_models = req_models = ['ST12000NM0007', 'ST4000DM000', 'ST8000NM0055', 'ST8000DM002']\n",
    "    df_final = pd.DataFrame()\n",
    "    for name in st_models:\n",
    "        full_name = name + number_of_days\n",
    "        dataset = pd.read_csv(\"../dataset/\" + full_name)\n",
    "        df = dataset[['model', 'capacity_bytes','serial_number', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']]\n",
    "        df_final = df_final.append(df, ignore_index=True)\n",
    "    return df_final"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\python\\python37\\lib\\site-packages\\ipykernel_launcher.py:4: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  after removing the cwd from sys.path.\n"
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
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_187_raw</th>\n",
       "      <th>smart_188_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_198_raw</th>\n",
       "      <th>capacity_bytes</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12000138625024</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12000138625024</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4809.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12000138625024</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12000138625024</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12000138625024</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86260</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8001563222016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86261</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8001563222016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86262</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8001563222016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86263</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8001563222016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86264</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8001563222016</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>86265 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       smart_5_raw  smart_187_raw  smart_188_raw  smart_197_raw  \\\n",
       "0              0.0            0.0            0.0            0.0   \n",
       "1              0.0            0.0            0.0            0.0   \n",
       "2           4809.0            1.0            0.0            0.0   \n",
       "3              0.0            0.0            0.0            0.0   \n",
       "4              0.0            0.0            0.0            0.0   \n",
       "...            ...            ...            ...            ...   \n",
       "86260          0.0            0.0            0.0            0.0   \n",
       "86261          0.0            0.0            0.0            0.0   \n",
       "86262          0.0            0.0            0.0            0.0   \n",
       "86263          0.0            0.0            0.0            0.0   \n",
       "86264          0.0            0.0            0.0            0.0   \n",
       "\n",
       "       smart_198_raw  capacity_bytes  \n",
       "0                0.0  12000138625024  \n",
       "1                0.0  12000138625024  \n",
       "2                0.0  12000138625024  \n",
       "3                0.0  12000138625024  \n",
       "4                0.0  12000138625024  \n",
       "...              ...             ...  \n",
       "86260            0.0   8001563222016  \n",
       "86261            0.0   8001563222016  \n",
       "86262            0.0   8001563222016  \n",
       "86263            0.0   8001563222016  \n",
       "86264            0.0   8001563222016  \n",
       "\n",
       "[86265 rows x 6 columns]"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "number_of_days=\"_last_day_all_q_raw.csv\"\n",
    "df_all_model=get_all_models(number_of_days)\n",
    "df_all_5f=df_all_model[smart_attributes]\n",
    "df_all_5f['capacity_bytes']=df_all_model['capacity_bytes']\n",
    "df_all_5f"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#without normalization\n",
    "plot_elbow(df_all_5f,'Elbow Method For Optimal k - only 5 features -without normalization all models')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "86265\n"
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
       "      <th>2</th>\n",
       "      <td>38210</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>46</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST4000DM000</th>\n",
       "      <th>1</th>\n",
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
       "ST12000NM0007 2               38210\n",
       "              3                  46\n",
       "ST4000DM000   1               23394\n",
       "ST8000DM002   0                9936\n",
       "ST8000NM0055  0               14679"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clustered_labels=run_kmeans(df_all_5f,4)\n",
    "print(len(clustered_labels))\n",
    "# display_result_frame(df_all_5f,clustered_labels)\n",
    "df_res = pd.DataFrame()\n",
    "df_res[\"serial_number\"] = df_all_model[\"serial_number\"] \n",
    "df_res[\"model\"] = df_all_model[\"model\"] \n",
    "df_res[\"labels\"] = clustered_labels\n",
    "df_res.groupby(['model', 'labels']).count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
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
       "      <th>capacity_bytes</th>\n",
       "      <th>serial_number</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_187_raw</th>\n",
       "      <th>smart_188_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_198_raw</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>model</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>ST12000NM0007</th>\n",
       "      <td>38256</td>\n",
       "      <td>38256</td>\n",
       "      <td>38256</td>\n",
       "      <td>38256</td>\n",
       "      <td>38256</td>\n",
       "      <td>38256</td>\n",
       "      <td>38256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST4000DM000</th>\n",
       "      <td>23394</td>\n",
       "      <td>23394</td>\n",
       "      <td>23394</td>\n",
       "      <td>23394</td>\n",
       "      <td>23394</td>\n",
       "      <td>23394</td>\n",
       "      <td>23394</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000DM002</th>\n",
       "      <td>9936</td>\n",
       "      <td>9936</td>\n",
       "      <td>9936</td>\n",
       "      <td>9936</td>\n",
       "      <td>9936</td>\n",
       "      <td>9936</td>\n",
       "      <td>9936</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000NM0055</th>\n",
       "      <td>14679</td>\n",
       "      <td>14679</td>\n",
       "      <td>14679</td>\n",
       "      <td>14679</td>\n",
       "      <td>14679</td>\n",
       "      <td>14679</td>\n",
       "      <td>14679</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               capacity_bytes  serial_number  smart_5_raw  smart_187_raw  \\\n",
       "model                                                                      \n",
       "ST12000NM0007           38256          38256        38256          38256   \n",
       "ST4000DM000             23394          23394        23394          23394   \n",
       "ST8000DM002              9936           9936         9936           9936   \n",
       "ST8000NM0055            14679          14679        14679          14679   \n",
       "\n",
       "               smart_188_raw  smart_197_raw  smart_198_raw  \n",
       "model                                                       \n",
       "ST12000NM0007          38256          38256          38256  \n",
       "ST4000DM000            23394          23394          23394  \n",
       "ST8000DM002             9936           9936           9936  \n",
       "ST8000NM0055           14679          14679          14679  "
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_all_model.groupby(['model']).count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data_transformed=normalize(df_all_5f)\n",
    "plot_elbow(data_transformed,'Elbow Method For Optimal k - only 5 SMART features -with normalization all models')"
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
       "      <th>0</th>\n",
       "      <td>38085</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>171</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST4000DM000</th>\n",
       "      <th>1</th>\n",
       "      <td>23394</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST8000DM002</th>\n",
       "      <th>2</th>\n",
       "      <td>9936</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">ST8000NM0055</th>\n",
       "      <th>2</th>\n",
       "      <td>14675</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      serial_number\n",
       "model         labels               \n",
       "ST12000NM0007 0               38085\n",
       "              3                 171\n",
       "ST4000DM000   1               23394\n",
       "ST8000DM002   2                9936\n",
       "ST8000NM0055  2               14675\n",
       "              3                   4"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clustered_labels=run_kmeans(data_transformed,4)\n",
    "# display_result_frame(df_all_5f,clustered_labels)\n",
    "df_res = pd.DataFrame()\n",
    "df_res[\"serial_number\"] = df_all_model[\"serial_number\"] \n",
    "df_res[\"model\"] = df_all_model[\"model\"] \n",
    "df_res[\"labels\"] = clustered_labels\n",
    "df_res.groupby(['model', 'labels']).count()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Trying PCA with capacity bytes:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x3580a97bc8>"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df_allf=data_transformed.copy()\n",
    "pca = PCA(n_components=2) #to visualise\n",
    "X = pca.fit_transform(df_allf)\n",
    "colors = {'ST12000NM0007':'red', 'ST4000DM000':'blue', 'ST8000DM002':'green', 'ST8000NM0055':'black'}\n",
    "\n",
    "# labels=df_all_model['model_label']\n",
    "labels=df_all_model['model']\n",
    "\n",
    "# plt.scatter(X[:,0],X[:,1],c=labels,label=labels, marker='x')\n",
    "plt.scatter(X[:,0],X[:,1], c=labels.apply(lambda x: colors[x]) , marker='x')\n",
    "plt.xlabel(\"feature 1\")\n",
    "plt.ylabel(\"feature 2\")\n",
    "plt.title(\"First 2 features from PCA (Capacity_bytes included)\")\n",
    "red = mpatches.Patch(color='red', label='ST12000NM0007')\n",
    "blue = mpatches.Patch(color='blue', label='ST4000DM000')\n",
    "green = mpatches.Patch(color='green', label='ST8000DM002')\n",
    "black = mpatches.Patch(color='black', label='ST8000NM0055')\n",
    "plt.legend(handles=[red,blue,green,black])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
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
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_187_raw</th>\n",
       "      <th>smart_188_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_198_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4809.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86260</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86261</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86262</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86263</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86264</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>86265 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       smart_5_raw  smart_187_raw  smart_188_raw  smart_197_raw  smart_198_raw\n",
       "0              0.0            0.0            0.0            0.0            0.0\n",
       "1              0.0            0.0            0.0            0.0            0.0\n",
       "2           4809.0            1.0            0.0            0.0            0.0\n",
       "3              0.0            0.0            0.0            0.0            0.0\n",
       "4              0.0            0.0            0.0            0.0            0.0\n",
       "...            ...            ...            ...            ...            ...\n",
       "86260          0.0            0.0            0.0            0.0            0.0\n",
       "86261          0.0            0.0            0.0            0.0            0.0\n",
       "86262          0.0            0.0            0.0            0.0            0.0\n",
       "86263          0.0            0.0            0.0            0.0            0.0\n",
       "86264          0.0            0.0            0.0            0.0            0.0\n",
       "\n",
       "[86265 rows x 5 columns]"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_no_cap=df_all_5f.drop(['capacity_bytes'],axis=1)\n",
    "df_no_cap #No capacity original data for all models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(86265, 2)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x3580ef7888>"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "norm_df_no_cap=normalize(df_no_cap)\n",
    "pca = PCA(n_components=2) #to visualise\n",
    "X = pca.fit_transform(norm_df_no_cap)\n",
    "print(X.shape)\n",
    "colors = {'ST12000NM0007':'red', 'ST4000DM000':'blue', 'ST8000DM002':'green', 'ST8000NM0055':'black'}\n",
    "\n",
    "# labels=df_all_model['model_label']\n",
    "labels=df_all_model['model']\n",
    "\n",
    "# plt.scatter(X[:,0],X[:,1],c=labels,label=labels, marker='x')\n",
    "plt.scatter(X[:,0],X[:,1], c=labels.apply(lambda x: colors[x]) , marker='x', label=['red','b','c','d'])\n",
    "plt.xlabel(\"feature 1\")\n",
    "plt.ylabel(\"feature 2\")\n",
    "plt.title(\"First 2 features from PCA (Capacity_bytes not included)\")\n",
    "red = mpatches.Patch(color='red', label='ST12000NM0007')\n",
    "blue = mpatches.Patch(color='blue', label='ST4000DM000')\n",
    "green = mpatches.Patch(color='green', label='ST8000DM002')\n",
    "black = mpatches.Patch(color='black', label='ST8000NM0055')\n",
    "plt.legend(handles=[red,blue,green,black])\n"
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
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
