{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_failed_disks( root = \"../dataset\", drive_file = \"/ST12000NM0007_last_10_day.csv\",  \n",
    "                          ignore_cols = [\"model\",\"capacity_bytes\",\"failure\"]):\n",
    "\n",
    "    df = pd.read_csv(root+drive_file, parse_dates=True, index_col=[0,1])\n",
    "\n",
    "    df_bad = df.loc[df['failure'] == 1]\n",
    "    df_bad = df_bad.sort_values([\"date\"])\n",
    "    bad_y = df_bad[\"failure\"]\n",
    "    df_bad.drop(columns=ignore_cols, inplace=True, axis=1)\n",
    "    return df_bad"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "def concat_failed(paths):\n",
    "    dfs = []\n",
    "    for path in paths: \n",
    "        dfs.append(get_failed_disks(drive_file=path))\n",
    "    all_dfs = pd.concat(dfs)\n",
    "    return all_dfs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Parth/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:5: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version\n",
      "of pandas will change to not sort by default.\n",
      "\n",
      "To accept the future behavior, pass 'sort=False'.\n",
      "\n",
      "To retain the current behavior and silence the warning, pass 'sort=True'.\n",
      "\n",
      "  \"\"\"\n"
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
       "      <th>smart_10_raw</th>\n",
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_183_raw</th>\n",
       "      <th>smart_184_raw</th>\n",
       "      <th>smart_187_raw</th>\n",
       "      <th>smart_188_raw</th>\n",
       "      <th>smart_189_raw</th>\n",
       "      <th>smart_190_raw</th>\n",
       "      <th>smart_191_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>...</th>\n",
       "      <th>smart_240_raw</th>\n",
       "      <th>smart_241_raw</th>\n",
       "      <th>smart_242_raw</th>\n",
       "      <th>smart_2_raw</th>\n",
       "      <th>smart_3_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_7_raw</th>\n",
       "      <th>smart_8_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>date</th>\n",
       "      <th>serial_number</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
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
       "      <td>2019-01-01</td>\n",
       "      <td>PL2331LAH9R55J</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>227.0</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>103.0</td>\n",
       "      <td>441.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>43.0</td>\n",
       "      <td>15774.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td rowspan=\"2\" valign=\"top\">2019-01-02</td>\n",
       "      <td>PL2331LAH3HXGJ</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>398.0</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>100.0</td>\n",
       "      <td>426.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>42.0</td>\n",
       "      <td>24108.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>PL2331LAH9R55J</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>228.0</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>103.0</td>\n",
       "      <td>441.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>43.0</td>\n",
       "      <td>15798.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td rowspan=\"2\" valign=\"top\">2019-01-03</td>\n",
       "      <td>PL2331LAH3HXGJ</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>399.0</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>100.0</td>\n",
       "      <td>426.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>42.0</td>\n",
       "      <td>24133.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>PL2331LAH9R55J</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>229.0</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>103.0</td>\n",
       "      <td>441.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>43.0</td>\n",
       "      <td>15822.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 30 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                           smart_10_raw  smart_12_raw  smart_183_raw  \\\n",
       "date       serial_number                                               \n",
       "2019-01-01 PL2331LAH9R55J           0.0           8.0            NaN   \n",
       "2019-01-02 PL2331LAH3HXGJ           0.0           9.0            NaN   \n",
       "           PL2331LAH9R55J           0.0           8.0            NaN   \n",
       "2019-01-03 PL2331LAH3HXGJ           0.0           9.0            NaN   \n",
       "           PL2331LAH9R55J           0.0           8.0            NaN   \n",
       "\n",
       "                           smart_184_raw  smart_187_raw  smart_188_raw  \\\n",
       "date       serial_number                                                 \n",
       "2019-01-01 PL2331LAH9R55J            NaN            NaN            NaN   \n",
       "2019-01-02 PL2331LAH3HXGJ            NaN            NaN            NaN   \n",
       "           PL2331LAH9R55J            NaN            NaN            NaN   \n",
       "2019-01-03 PL2331LAH3HXGJ            NaN            NaN            NaN   \n",
       "           PL2331LAH9R55J            NaN            NaN            NaN   \n",
       "\n",
       "                           smart_189_raw  smart_190_raw  smart_191_raw  \\\n",
       "date       serial_number                                                 \n",
       "2019-01-01 PL2331LAH9R55J            NaN            NaN            NaN   \n",
       "2019-01-02 PL2331LAH3HXGJ            NaN            NaN            NaN   \n",
       "           PL2331LAH9R55J            NaN            NaN            NaN   \n",
       "2019-01-03 PL2331LAH3HXGJ            NaN            NaN            NaN   \n",
       "           PL2331LAH9R55J            NaN            NaN            NaN   \n",
       "\n",
       "                           smart_192_raw  ...  smart_240_raw  smart_241_raw  \\\n",
       "date       serial_number                  ...                                 \n",
       "2019-01-01 PL2331LAH9R55J          227.0  ...            NaN            NaN   \n",
       "2019-01-02 PL2331LAH3HXGJ          398.0  ...            NaN            NaN   \n",
       "           PL2331LAH9R55J          228.0  ...            NaN            NaN   \n",
       "2019-01-03 PL2331LAH3HXGJ          399.0  ...            NaN            NaN   \n",
       "           PL2331LAH9R55J          229.0  ...            NaN            NaN   \n",
       "\n",
       "                           smart_242_raw  smart_2_raw  smart_3_raw  \\\n",
       "date       serial_number                                             \n",
       "2019-01-01 PL2331LAH9R55J            NaN        103.0        441.0   \n",
       "2019-01-02 PL2331LAH3HXGJ            NaN        100.0        426.0   \n",
       "           PL2331LAH9R55J            NaN        103.0        441.0   \n",
       "2019-01-03 PL2331LAH3HXGJ            NaN        100.0        426.0   \n",
       "           PL2331LAH9R55J            NaN        103.0        441.0   \n",
       "\n",
       "                           smart_4_raw  smart_5_raw  smart_7_raw  smart_8_raw  \\\n",
       "date       serial_number                                                        \n",
       "2019-01-01 PL2331LAH9R55J          8.0         20.0          0.0         43.0   \n",
       "2019-01-02 PL2331LAH3HXGJ          9.0          0.0          0.0         42.0   \n",
       "           PL2331LAH9R55J          8.0         20.0          0.0         43.0   \n",
       "2019-01-03 PL2331LAH3HXGJ          9.0          0.0          0.0         42.0   \n",
       "           PL2331LAH9R55J          8.0         20.0          0.0         43.0   \n",
       "\n",
       "                           smart_9_raw  \n",
       "date       serial_number                \n",
       "2019-01-01 PL2331LAH9R55J      15774.0  \n",
       "2019-01-02 PL2331LAH3HXGJ      24108.0  \n",
       "           PL2331LAH9R55J      15798.0  \n",
       "2019-01-03 PL2331LAH3HXGJ      24133.0  \n",
       "           PL2331LAH9R55J      15822.0  \n",
       "\n",
       "[5 rows x 30 columns]"
      ]
     },
     "execution_count": 121,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "paths = [\n",
    "    \"/HGST HMS5C4040BLE640_last_10_day_all_q_raw.csv\",\n",
    "    \"/ST4000DM000_last_10_day_all_q_raw.csv\",\n",
    "    \"/ST8000DM002_last_10_day_all_q_raw.csv\",\n",
    "    \"/ST12000NM0007_last_10_day_all_q_raw.csv\",\n",
    "    \"/TOSHIBA MQ01ABF050_last_10_day_all_q_raw.csv\",\n",
    "    \"/ST8000NM0055_last_10_day_all_q_raw.csv\"\n",
    "]\n",
    "\n",
    "failed_disks = concat_failed(paths)\n",
    "failed_disks.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>date</th>\n",
       "      <th>serial_number</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
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
       "      <td>2019-01-01</td>\n",
       "      <td>PL2331LAH9R55J</td>\n",
       "      <td>8.0</td>\n",
       "      <td>227.0</td>\n",
       "      <td>227.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>56.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>15774.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td rowspan=\"2\" valign=\"top\">2019-01-02</td>\n",
       "      <td>PL2331LAH3HXGJ</td>\n",
       "      <td>9.0</td>\n",
       "      <td>398.0</td>\n",
       "      <td>398.0</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>24108.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>PL2331LAH9R55J</td>\n",
       "      <td>8.0</td>\n",
       "      <td>228.0</td>\n",
       "      <td>228.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>56.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>15798.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td rowspan=\"2\" valign=\"top\">2019-01-03</td>\n",
       "      <td>PL2331LAH3HXGJ</td>\n",
       "      <td>9.0</td>\n",
       "      <td>399.0</td>\n",
       "      <td>399.0</td>\n",
       "      <td>31.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>24133.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>PL2331LAH9R55J</td>\n",
       "      <td>8.0</td>\n",
       "      <td>229.0</td>\n",
       "      <td>229.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>56.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>15822.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                           smart_12_raw  smart_192_raw  smart_193_raw  \\\n",
       "date       serial_number                                                \n",
       "2019-01-01 PL2331LAH9R55J           8.0          227.0          227.0   \n",
       "2019-01-02 PL2331LAH3HXGJ           9.0          398.0          398.0   \n",
       "           PL2331LAH9R55J           8.0          228.0          228.0   \n",
       "2019-01-03 PL2331LAH3HXGJ           9.0          399.0          399.0   \n",
       "           PL2331LAH9R55J           8.0          229.0          229.0   \n",
       "\n",
       "                           smart_194_raw  smart_197_raw  smart_199_raw  \\\n",
       "date       serial_number                                                 \n",
       "2019-01-01 PL2331LAH9R55J           35.0           56.0            0.0   \n",
       "2019-01-02 PL2331LAH3HXGJ           32.0            0.0            0.0   \n",
       "           PL2331LAH9R55J           35.0           56.0            0.0   \n",
       "2019-01-03 PL2331LAH3HXGJ           31.0            0.0            0.0   \n",
       "           PL2331LAH9R55J           35.0           56.0            0.0   \n",
       "\n",
       "                           smart_4_raw  smart_5_raw  smart_9_raw  \n",
       "date       serial_number                                          \n",
       "2019-01-01 PL2331LAH9R55J          8.0         20.0      15774.0  \n",
       "2019-01-02 PL2331LAH3HXGJ          9.0          0.0      24108.0  \n",
       "           PL2331LAH9R55J          8.0         20.0      15798.0  \n",
       "2019-01-03 PL2331LAH3HXGJ          9.0          0.0      24133.0  \n",
       "           PL2331LAH9R55J          8.0         20.0      15822.0  "
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "failed_disks.dropna(axis=1, inplace=True)\n",
    "failed_disks.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>date</th>\n",
       "      <th>serial_number</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
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
       "      <td>2019-01-01</td>\n",
       "      <td>PL2331LAH9R55J</td>\n",
       "      <td>8.0</td>\n",
       "      <td>227.0</td>\n",
       "      <td>227.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>56.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>15774.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td rowspan=\"2\" valign=\"top\">2019-01-02</td>\n",
       "      <td>PL2331LAH3HXGJ</td>\n",
       "      <td>9.0</td>\n",
       "      <td>398.0</td>\n",
       "      <td>398.0</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>24108.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>PL2331LAH9R55J</td>\n",
       "      <td>8.0</td>\n",
       "      <td>228.0</td>\n",
       "      <td>228.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>56.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>15798.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td rowspan=\"2\" valign=\"top\">2019-01-03</td>\n",
       "      <td>PL2331LAH3HXGJ</td>\n",
       "      <td>9.0</td>\n",
       "      <td>399.0</td>\n",
       "      <td>399.0</td>\n",
       "      <td>31.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>24133.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>PL2331LAH9R55J</td>\n",
       "      <td>8.0</td>\n",
       "      <td>229.0</td>\n",
       "      <td>229.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>56.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>15822.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                           smart_12_raw  smart_192_raw  smart_193_raw  \\\n",
       "date       serial_number                                                \n",
       "2019-01-01 PL2331LAH9R55J           8.0          227.0          227.0   \n",
       "2019-01-02 PL2331LAH3HXGJ           9.0          398.0          398.0   \n",
       "           PL2331LAH9R55J           8.0          228.0          228.0   \n",
       "2019-01-03 PL2331LAH3HXGJ           9.0          399.0          399.0   \n",
       "           PL2331LAH9R55J           8.0          229.0          229.0   \n",
       "\n",
       "                           smart_194_raw  smart_197_raw  smart_199_raw  \\\n",
       "date       serial_number                                                 \n",
       "2019-01-01 PL2331LAH9R55J           35.0           56.0            0.0   \n",
       "2019-01-02 PL2331LAH3HXGJ           32.0            0.0            0.0   \n",
       "           PL2331LAH9R55J           35.0           56.0            0.0   \n",
       "2019-01-03 PL2331LAH3HXGJ           31.0            0.0            0.0   \n",
       "           PL2331LAH9R55J           35.0           56.0            0.0   \n",
       "\n",
       "                           smart_4_raw  smart_5_raw  smart_9_raw  \n",
       "date       serial_number                                          \n",
       "2019-01-01 PL2331LAH9R55J          8.0         20.0      15774.0  \n",
       "2019-01-02 PL2331LAH3HXGJ          9.0          0.0      24108.0  \n",
       "           PL2331LAH9R55J          8.0         20.0      15798.0  \n",
       "2019-01-03 PL2331LAH3HXGJ          9.0          0.0      24133.0  \n",
       "           PL2331LAH9R55J          8.0         20.0      15822.0  "
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nunique = failed_disks.apply(pd.Series.nunique)\n",
    "cols_to_drop = nunique[nunique == 1].index\n",
    "failed_disks.drop(cols_to_drop, axis=1, inplace=True)\n",
    "failed_disks.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                           0  1\n",
      "0      (2019-01-01 00:00:00, PL2331LAH9R55J)  4\n",
      "1      (2019-01-02 00:00:00, PL2331LAH3HXGJ)  4\n",
      "2      (2019-01-02 00:00:00, PL2331LAH9R55J)  4\n",
      "3      (2019-01-03 00:00:00, PL2331LAH3HXGJ)  4\n",
      "4      (2019-01-03 00:00:00, PL2331LAH9R55J)  4\n",
      "...                                      ... ..\n",
      "19929        (2019-12-27 00:00:00, ZA180ZVF)  3\n",
      "19930        (2019-12-28 00:00:00, ZA17WKQ0)  3\n",
      "19931        (2019-12-28 00:00:00, ZA1815CF)  3\n",
      "19932        (2019-12-29 00:00:00, ZA17WKQ0)  0\n",
      "19933        (2019-12-30 00:00:00, ZA17WKQ0)  0\n",
      "\n",
      "[19934 rows x 2 columns]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.cluster import KMeans\n",
    "\n",
    "mat = failed_disks.values\n",
    "kmeans = KMeans(n_clusters=5, random_state=0).fit(mat)\n",
    "# Get cluster assignment labels\n",
    "labels = kmeans.labels_\n",
    "# Format results as a DataFrame\n",
    "results = pd.DataFrame([failed_disks.index,labels]).T\n",
    "print(results)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4], dtype=int32)"
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.unique(kmeans.labels_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "572"
      ]
     },
     "execution_count": 126,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(results[results[1] == 0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "360"
      ]
     },
     "execution_count": 127,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(results[results[1] == 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "30"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(results[results[1] == 2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5378"
      ]
     },
     "execution_count": 129,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(results[results[1] == 3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "13594"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(results[results[1] == 4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_model_nos(cluster_no):\n",
    "    cluster = results[results[1] == cluster_no]\n",
    "    model_nos = []\n",
    "    for key in cluster[0]:\n",
    "        model_no = key[1]\n",
    "        model_nos.append(model_no)\n",
    "    return set(model_nos)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_nos = []\n",
    "for i in range(5):\n",
    "    model_nos.append(get_model_nos(i))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Common in cluster  0  cluster 1 :  0\n",
      "Common in cluster  0  cluster 2 :  0\n",
      "Common in cluster  0  cluster 3 :  7\n",
      "Common in cluster  0  cluster 4 :  100\n",
      "Common in cluster  1  cluster 2 :  0\n",
      "Common in cluster  1  cluster 3 :  0\n",
      "Common in cluster  1  cluster 4 :  0\n",
      "Common in cluster  2  cluster 3 :  0\n",
      "Common in cluster  2  cluster 4 :  0\n",
      "Common in cluster  3  cluster 4 :  5\n"
     ]
    }
   ],
   "source": [
    "for i in range(5):\n",
    "    for j in range(i+1, 5):\n",
    "        set_i = model_nos[i]\n",
    "        set_j = model_nos[j]\n",
    "        print(\"Common in cluster \", i, \" cluster\", j, \": \", len(set_i.intersection(set_j)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {
    "scrolled": true
   },
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>6.861888</td>\n",
       "      <td>181.335664</td>\n",
       "      <td>2897.019231</td>\n",
       "      <td>29.342657</td>\n",
       "      <td>9.86014</td>\n",
       "      <td>0.031469</td>\n",
       "      <td>9.108392</td>\n",
       "      <td>36452.048951</td>\n",
       "      <td>15016.346154</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   smart_12_raw  smart_192_raw  smart_193_raw  smart_194_raw  smart_197_raw  \\\n",
       "0      6.861888     181.335664    2897.019231      29.342657        9.86014   \n",
       "\n",
       "   smart_199_raw  smart_4_raw   smart_5_raw   smart_9_raw  \n",
       "0       0.031469     9.108392  36452.048951  15016.346154  "
      ]
     },
     "metadata": {},
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>4.592328</td>\n",
       "      <td>78.803037</td>\n",
       "      <td>5120.262818</td>\n",
       "      <td>6.353404</td>\n",
       "      <td>100.504967</td>\n",
       "      <td>0.249109</td>\n",
       "      <td>11.260743</td>\n",
       "      <td>13386.13968</td>\n",
       "      <td>4044.094035</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   smart_12_raw  smart_192_raw  smart_193_raw  smart_194_raw  smart_197_raw  \\\n",
       "0      4.592328      78.803037    5120.262818       6.353404     100.504967   \n",
       "\n",
       "   smart_199_raw  smart_4_raw  smart_5_raw  smart_9_raw  \n",
       "0       0.249109    11.260743  13386.13968  4044.094035  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "idx = results[results[1] == 0][0].values\n",
    "cluster = failed_disks.loc[idx]\n",
    "mean = cluster.mean().to_frame()\n",
    "display(mean.transpose())\n",
    "std = cluster.std().to_frame()\n",
    "display(std.transpose())"
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
   "execution_count": 135,
   "metadata": {
    "scrolled": true
   },
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>11.755556</td>\n",
       "      <td>0.416667</td>\n",
       "      <td>162295.675</td>\n",
       "      <td>22.302778</td>\n",
       "      <td>40.0</td>\n",
       "      <td>1.444444</td>\n",
       "      <td>12.672222</td>\n",
       "      <td>60.755556</td>\n",
       "      <td>32921.15</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   smart_12_raw  smart_192_raw  smart_193_raw  smart_194_raw  smart_197_raw  \\\n",
       "0     11.755556       0.416667     162295.675      22.302778           40.0   \n",
       "\n",
       "   smart_199_raw  smart_4_raw  smart_5_raw  smart_9_raw  \n",
       "0       1.444444    12.672222    60.755556     32921.15  "
      ]
     },
     "metadata": {},
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>7.736339</td>\n",
       "      <td>1.234937</td>\n",
       "      <td>41887.118687</td>\n",
       "      <td>3.366225</td>\n",
       "      <td>168.891652</td>\n",
       "      <td>8.064657</td>\n",
       "      <td>7.087853</td>\n",
       "      <td>341.047847</td>\n",
       "      <td>3186.692562</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   smart_12_raw  smart_192_raw  smart_193_raw  smart_194_raw  smart_197_raw  \\\n",
       "0      7.736339       1.234937   41887.118687       3.366225     168.891652   \n",
       "\n",
       "   smart_199_raw  smart_4_raw  smart_5_raw  smart_9_raw  \n",
       "0       8.064657     7.087853   341.047847  3186.692562  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "idx = results[results[1] == 1][0].values\n",
    "cluster = failed_disks.loc[idx]\n",
    "mean = cluster.mean().to_frame()\n",
    "display(mean.transpose())\n",
    "std = cluster.std().to_frame()\n",
    "display(std.transpose())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#197 Current pending sector count\n",
    "#5 Reallocated sector count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {
    "scrolled": true
   },
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>17.666667</td>\n",
       "      <td>1.333333</td>\n",
       "      <td>436522.033333</td>\n",
       "      <td>28.366667</td>\n",
       "      <td>224.8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18.0</td>\n",
       "      <td>60.533333</td>\n",
       "      <td>34216.166667</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   smart_12_raw  smart_192_raw  smart_193_raw  smart_194_raw  smart_197_raw  \\\n",
       "0     17.666667       1.333333  436522.033333      28.366667          224.8   \n",
       "\n",
       "   smart_199_raw  smart_4_raw  smart_5_raw   smart_9_raw  \n",
       "0            0.0         18.0    60.533333  34216.166667  "
      ]
     },
     "metadata": {},
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>3.924576</td>\n",
       "      <td>1.268541</td>\n",
       "      <td>73612.634184</td>\n",
       "      <td>1.564329</td>\n",
       "      <td>1192.09229</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.805622</td>\n",
       "      <td>331.554721</td>\n",
       "      <td>3444.378089</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   smart_12_raw  smart_192_raw  smart_193_raw  smart_194_raw  smart_197_raw  \\\n",
       "0      3.924576       1.268541   73612.634184       1.564329     1192.09229   \n",
       "\n",
       "   smart_199_raw  smart_4_raw  smart_5_raw  smart_9_raw  \n",
       "0            0.0     3.805622   331.554721  3444.378089  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "idx = results[results[1] == 2][0].values\n",
    "cluster = failed_disks.loc[idx]\n",
    "mean = cluster.mean().to_frame()\n",
    "display(mean.transpose())\n",
    "std = cluster.std().to_frame()\n",
    "display(std.transpose())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#12 Power cycle count\n",
    "#192 Power-off retract count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>8.243585</td>\n",
       "      <td>53.828003</td>\n",
       "      <td>18446.274266</td>\n",
       "      <td>27.302901</td>\n",
       "      <td>34.587951</td>\n",
       "      <td>1.343994</td>\n",
       "      <td>8.503719</td>\n",
       "      <td>528.892711</td>\n",
       "      <td>28770.666605</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   smart_12_raw  smart_192_raw  smart_193_raw  smart_194_raw  smart_197_raw  \\\n",
       "0      8.243585      53.828003   18446.274266      27.302901      34.587951   \n",
       "\n",
       "   smart_199_raw  smart_4_raw  smart_5_raw   smart_9_raw  \n",
       "0       1.343994     8.503719   528.892711  28770.666605  "
      ]
     },
     "metadata": {},
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>5.484163</td>\n",
       "      <td>1124.201848</td>\n",
       "      <td>12629.042302</td>\n",
       "      <td>7.048249</td>\n",
       "      <td>958.674745</td>\n",
       "      <td>16.656345</td>\n",
       "      <td>6.619559</td>\n",
       "      <td>2293.114454</td>\n",
       "      <td>6603.94135</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   smart_12_raw  smart_192_raw  smart_193_raw  smart_194_raw  smart_197_raw  \\\n",
       "0      5.484163    1124.201848   12629.042302       7.048249     958.674745   \n",
       "\n",
       "   smart_199_raw  smart_4_raw  smart_5_raw  smart_9_raw  \n",
       "0      16.656345     6.619559  2293.114454   6603.94135  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "idx = results[results[1] == 3][0].values\n",
    "cluster = failed_disks.loc[idx]\n",
    "mean = cluster.mean().to_frame()\n",
    "display(mean.transpose())\n",
    "std = cluster.std().to_frame()\n",
    "display(std.transpose())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#199 UDMA CRC error rate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {
    "scrolled": true
   },
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>5.920995</td>\n",
       "      <td>145.032956</td>\n",
       "      <td>1959.928719</td>\n",
       "      <td>30.060174</td>\n",
       "      <td>19.070767</td>\n",
       "      <td>1.471311</td>\n",
       "      <td>12.784537</td>\n",
       "      <td>1147.917905</td>\n",
       "      <td>12905.066794</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   smart_12_raw  smart_192_raw  smart_193_raw  smart_194_raw  smart_197_raw  \\\n",
       "0      5.920995     145.032956    1959.928719      30.060174      19.070767   \n",
       "\n",
       "   smart_199_raw  smart_4_raw  smart_5_raw   smart_9_raw  \n",
       "0       1.471311    12.784537  1147.917905  12905.066794  "
      ]
     },
     "metadata": {},
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
       "      <th>smart_12_raw</th>\n",
       "      <th>smart_192_raw</th>\n",
       "      <th>smart_193_raw</th>\n",
       "      <th>smart_194_raw</th>\n",
       "      <th>smart_197_raw</th>\n",
       "      <th>smart_199_raw</th>\n",
       "      <th>smart_4_raw</th>\n",
       "      <th>smart_5_raw</th>\n",
       "      <th>smart_9_raw</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>4.732459</td>\n",
       "      <td>147.821046</td>\n",
       "      <td>2794.973668</td>\n",
       "      <td>6.464685</td>\n",
       "      <td>174.871998</td>\n",
       "      <td>37.733175</td>\n",
       "      <td>170.353086</td>\n",
       "      <td>3003.026049</td>\n",
       "      <td>5473.36484</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   smart_12_raw  smart_192_raw  smart_193_raw  smart_194_raw  smart_197_raw  \\\n",
       "0      4.732459     147.821046    2794.973668       6.464685     174.871998   \n",
       "\n",
       "   smart_199_raw  smart_4_raw  smart_5_raw  smart_9_raw  \n",
       "0      37.733175   170.353086  3003.026049   5473.36484  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "idx = results[results[1] == 4][0].values\n",
    "cluster = failed_disks.loc[idx]\n",
    "mean = cluster.mean().to_frame()\n",
    "display(mean.transpose())\n",
    "std = cluster.std().to_frame()\n",
    "display(std.transpose())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#193 Load/Unload cycle count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "133\n",
      "{'ZJV03NCN', 'ZCH06HG5', 'ZCH0DG4V', 'ZCH077N9', 'ZCH0995N', 'ZJV03KM5', 'ZCH0CDHE', 'ZJV02CR4', 'ZCH07ZYJ', 'ZJV0WBV6', 'ZA17WKQ0', 'ZCH0BMET', 'ZCH0B4YA', 'ZJV00DR7', 'ZJV03NZ1', 'ZJV12R9J', 'ZJV00FRS', 'ZCH07NZ5', 'ZA180XTV', 'ZCH06VQE', 'ZCH07M4C', 'ZCH06DX3', 'ZJV03JEL', 'ZCH07F30', 'ZCH0CYQR', 'ZA189XNN', 'ZCH073TG', 'ZCH06VE2', 'ZA16DDRG', 'ZCH037K5', 'Z3054R08', 'ZCH07ZDK', 'ZCH06KQY', 'ZJV0WEQM', 'ZA18197Z', 'ZCH0AD9K', 'ZA17ZLZE', 'ZCH06GPN', 'ZCH07S1G', 'ZCH091PJ', 'ZCH08XHJ', 'ZCH07Y48', 'ZJV03EXA', 'ZCH0DG3H', 'ZCH07Q3L', 'ZCH07BXA', 'ZCH06WPN', 'ZCH07RJM', 'ZJV0SWXA', 'ZCH06X4H', 'ZJV0EB23', 'ZCH0BWPG', 'ZCH068EH', 'ZCH0AH5N', 'ZA14EK1F', 'ZCH07RYZ', 'ZCH0B3M5', 'ZCH09MHJ', 'ZCH07SKQ', 'ZCH07FQK', 'ZJV0D8LY', 'ZCH06FFA', 'ZCH07BWD', 'ZCH09GZ3', 'ZCH09WZZ', 'ZCH0A24Q', 'ZCH07H70', 'ZJV0ZW66', 'ZCH0CES4', 'ZCH0A83J', 'ZCH0CCKJ', 'ZCH07WD0', 'ZCH080AD', 'ZCH07HKF', 'ZCH01CPY', 'ZJV008TM', 'ZCH0C02G', 'ZJV0CF01', 'ZCH0B0SJ', 'ZCH08VPK', 'ZCH0D1HC', 'ZJV1CS48', 'ZJV0KB9Z', 'ZCH08A2N', 'ZCH0CC3Q', 'ZCH0BL6P', 'ZCH0CY2A', 'ZCH0BSBX', 'ZCH07WGZ', 'ZA18109B', 'Z304JVY9', 'ZCH07JQX', 'ZJV0WCXQ', 'ZCH086XE', 'ZCH0CXS2', 'ZCH06L67', 'ZA16DSXP', 'ZCH0A3FM', 'ZCH0926P', 'ZCH07ZRN', 'ZJV00FFT', 'ZCH0CZRM', 'ZJV1CGL8', 'ZA11SBWJ', 'ZCH07ERS', 'ZCH0A2DH', 'ZCH0A2Y7', 'ZJV1C49G', 'ZA18A89Q', 'ZCH07WBW', 'ZCH0CE31', 'ZCH05HMS', 'ZCH044NK', 'ZCH06J46', 'ZCH07D8D', 'ZCH06ZR3', 'ZJV005M5', 'ZCH07HYF', 'ZJV0016J', 'ZCH0AQ0J', 'ZCH0784T', 'ZCH06J49', 'ZJV00GZF', 'ZCH08BSC', 'ZCH0D1FB', 'ZCH06F61', 'ZJV03W5T', 'ZCH06H3V', 'ZJV00FQZ', 'ZCH05BD5', 'ZCH07HM0', 'ZJV0WFMV', 'ZCH0CALM'}\n"
     ]
    }
   ],
   "source": [
    "set0 = set([i[1] for i in results[results[1] == 0][0].values]) \n",
    "print(len(set0))\n",
    "print(str(set0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "36\n",
      "{'S300XB04', 'S3010D9D', 'S300VLA2', 'Z304KCCF', 'Z302T6F4', 'Z3051DGR', 'Z3025K5H', 'Z304YF91', 'Z304HV08', 'Z304HVC0', 'S301P0ES', 'Z304JPSZ', 'S301GM66', 'S300XAP9', 'S300Z731', 'Z304JP7M', 'Z304JZG7', 'Z304H823', 'Z3041TE5', 'Z304HSVA', 'Z304HSCT', 'Z30251SR', 'S300WGHB', 'Z303R1DE', 'Z303Y470', 'S300YPK7', 'Z3025M1G', 'Z3039XWR', 'S301PSAT', 'S300XCAB', 'Z304WT4G', 'S301P21G', 'S301NH5Z', 'Z3058XP6', 'Z302T6A8', 'Z30251BE'}\n"
     ]
    }
   ],
   "source": [
    "set1 = set([i[1] for i in results[results[1] == 1][0].values]) \n",
    "print(len(set1))\n",
    "print(str(set1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3\n",
      "{'Z3025LXD', 'Z30251XR', 'Z302CYHA'}\n"
     ]
    }
   ],
   "source": [
    "set2 = set([i[1] for i in results[results[1] == 2][0].values]) \n",
    "print(len(set2))\n",
    "print(str(set2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "549\n",
      "{'Z304KCR9', 'ZA180QNJ', 'ZA13R4G5', '96ICTBQOT', 'Z305D51Q', 'Z304HRM4', 'Z304JS04', 'S300Z6M5', 'S300Z8P7', 'Z302T9AH', 'ZA11X0CF', 'ZA16DY1V', 'ZA1421XC', 'ZA1349GR', 'ZA122YFE', 'ZA189JX6', 'Z302SYJA', 'ZA12JTD2', 'ZA189XP8', 'S301NKJR', 'Z304JHF7', 'S301GQG3', 'Z304JG10', 'Z304JP0J', 'Z304J2H9', 'ZA180XTV', 'Z302SZP3', 'S301GQV4', 'ZA189K76', 'ZA171VE9', 'S301NK5Y', 'Z305CXS7', 'Z302SLDV', 'S301GRL1', 'ZA12Q641', 'S301GQX5', 'S301GMK4', 'S301K0XF', 'ZA18166B', 'Z304L8JW', 'Z302SYMP', 'Z305GXJW', 'Z305GXC1', 'S300ZREZ', 'S3011818', 'S301GPB1', 'ZA1819HV', 'Z304XN5C', 'Z302SWXD', 'Z302F2WA', 'Z302AB82', 'Z302T6JA', 'ZA13R2GE', 'Z302G15C', 'ZA1818BX', 'ZA180ZVF', 'Z304TK95', 'Z305AXS8', 'ZA131WTR', 'ZA126J4M', 'ZA14ECR7', 'Z304JG8D', 'Z305KW12', 'ZA13WWFB', 'Z3026Q5Y', 'Z304HV83', 'S301GP34', 'S301GM8H', 'Z304K4BB', 'Z304JJ0A', 'ZA16CXNA', 'ZA189S3R', 'Z304JSVS', 'ZA189MPQ', 'ZA1815XB', 'Z304JGSZ', 'S301GMEP', 'Z3059CCB', 'ZA13R3AG', 'Z302T7HK', 'Z302T7X4', 'ZA180ZHC', 'ZA13YLAQ', 'ZA18BLN9', 'Z304HWJF', 'Z304JMK8', 'Z304HSCX', 'S300Z43B', '66HLS2P8S', 'ZA142SAB', 'S301168X', 'Z305D66M', 'S301GNWY', 'ZA13ZS1C', 'Z305K2J0', 'Z304JDL5', 'ZA142J6D', 'ZA13AG2K', 'ZA10V1N6', 'Z302B0NP', 'Z304JD9J', 'Z302T6CW', 'S301PSFV', 'Z302SZJH', 'S301GRGY', 'Z30271N2', 'S300Z60X', 'S3010M10', 'Z30513PV', 'ZA180YF7', 'Z304JMA0', 'S300Z43Q', 'S300Z4HB', 'ZA1438RR', 'ZA11RRR8', 'ZA12QFMF', 'PL2331LAHDNGXJ', 'Z304J0NT', 'ZA143CQS', 'Z304TDET', 'ZA13EWB5', 'S301GMBN', 'Z303QPVA', 'Z304JPNA', 'ZA13ZS5A', 'ZA18AB4N', 'PL2331LAGMKT0J', 'ZA13ECZ9', 'S301P6Y6', 'Z304JG8K', 'ZA189JVA', 'S300Z5S4', 'Z302AKLA', 'S300ZRP2', 'ZA13R3J8', 'S30115AC', 'ZA11X0FD', 'Z302T88H', 'Z302T6JR', 'Z305GW1E', 'S301NGZN', 'ZA13EWQP', '96ICTBQGT', 'Z304KCQR', 'ZA13R24Q', 'ZA13R2NE', 'ZA13F75G', 'ZA11V2AP', 'ZA16DV86', 'ZA11NHC2', 'S301LN54', 'Z304LA21', 'Z304JVZY', 'S301GQ7K', 'ZA189HT4', 'ZA12Q6BV', '57HBWEEKT', 'Z302T8LB', 'Z303MZ9Y', 'Z302BV7C', 'Z304J0TR', 'Z304J0QA', 'ZA14ED84', 'ZA181HSL', 'Z302ALD9', 'S301GP0D', 'ZA16ZZ7V', 'S300YSGX', 'Z3040DW9', 'Z305KWFK', 'Z304HPX7', 'ZA180RDR', 'Z304V9DH', 'Z305D4L5', 'Z304HS1X', 'S300Z8CZ', 'ZA13YLQ9', 'ZA171SGA', 'ZA14EK1F', 'ZA154WKK', 'Z302SYHF', 'ZA11X9WD', 'Z302SW4B', 'S300Z5JP', 'Z305D68P', 'Z304HSN8', 'ZA120GQQ', 'S301F73K', 'Z3041NA6', 'Z305D21E', 'S301GMCV', 'Z304KBKG', 'S301M4YT', 'Z303N1QB', 'ZA12HANQ', 'S300YTAY', 'Z304LAEN', 'ZA12JTF2', 'Z300VXA5', 'S30115TG', 'ZA11X9W0', 'Z305KVHS', 'ZA13QZP8', 'Z302F381', 'Z302SZHM', 'ZA13R8A9', 'ZA128XR7', 'ZA14EVX7', 'Z304ZBFY', 'Z302T6BL', 'Z304VW9S', 'ZA1815CF', 'Z304JVY9', 'Z304KCPF', 'S301GNMZ', 'S300Z9A6', 'S300WDA3', 'ZA16CLR7', 'Z302S7HN', 'ZA11LZSP', 'ZA12KEDM', 'Z3052BGS', 'Z302SZ6R', 'S301NKQV', 'ZA16DSXV', 'ZA12KYXV', 'Z304L9AJ', 'Z304JM7Q', 'Z302SXL0', 'Z3040DVF', 'ZA12MEX5', 'Z302AL1L', 'ZA11SBWJ', 'ZA180YMP', 'Z302SR45', 'Z305GWDQ', 'ZA11FJNG', 'S301GQ54', 'Z302T68R', 'ZA12Q5WX', '17OYTGKYT', 'Z304HGW3', '96ICTBQ0T', 'ZA18BV0E', 'S301NL7L', 'ZA13Z8EY', 'Z304JNLW', 'Z305L695', 'Z304RKYC', 'Z3041NJE', 'Z304JZH6', 'ZA13YP0A', 'ZA13YX8T', 'Z305NZY4', 'Z304JZK4', 'S301GP8P', 'ZA13QBVZ', 'Z304T5E0', 'ZA14ESPP', 'ZA154VHQ', 'Z300YLXG', 'Z304HSX5', 'Z3027HSX', 'ZA12RAWL', 'Z3029GTD', 'ZA1814T9', 'ZA171V2E', 'ZA16DQNH', 'ZA189WWC', 'S301GNWB', 'Z305D56S', 'Z302701J', 'Z304JW84', 'Z304KX6Y', '564ESIYGS', 'Z302T88S', 'ZA135ZLD', 'ZA1814L2', 'Z302SQJ9', 'S301NH71', 'ZA16DQA6', 'S30115HR', 'Z305DVWM', 'Z304L9AM', 'ZA130MBP', 'Z305DVQ7', 'ZA10YJFX', 'ZA143STV', 'S301KX89', 'S300Z5G1', 'ZA154E8D', 'ZA11LP0X', 'S300Z4YH', 'Z3040Z4M', 'S301GQZD', 'S300XE94', 'Z304HP4Z', 'Z3029FAN', 'Z303RRM9', '17OYTGL2T', 'Z304JN7J', 'Z305DW3N', 'S301GND9', 'Z304JVYJ', 'Z3059KNJ', 'Z3029AR3', 'Z304JLZW', 'Z302SZLQ', '17OYTGL8T', 'Z304V9Z7', 'Z304JWPG', 'ZA17X8B7', 'ZA189Y1M', 'ZA13R4LE', 'Z304KD5E', 'ZA1554BL', 'Z302EQBN', 'Z304L856', 'Z304X7VV', 'ZA154W5T', 'Z304KABR', 'Z305D5L0', 'S301GR6X', 'ZA171RQB', 'ZA13QBZ2', 'Z302712G', 'S301K5QN', 'Z304JLCE', 'ZA140NHA', 'Z30555PB', 'ZA16DQM6', 'ZA11J9ZZ', 'Z3053NZ4', 'ZA157XW5', 'ZA12KEHD', 'ZA126E56', 'Z304KR1N', 'S301GQSM', 'Z304JMZP', 'ZA13Z8GC', 'S301K0GV', 'Z305D6G2', 'ZA13Z44R', 'Z302T6LE', 'S301GQQV', 'S301GT3N', 'Z302SX25', 'Z304JDDH', 'ZA181JK1', 'S300ZQVL', 'ZA13QZP2', 'S30108RC', 'ZA121C0M', 'ZA12RB1D', 'Z3029GKK', 'ZA13R2EL', 'Z305B5X2', 'S301GQVW', 'S301167F', 'ZA13GZB3', 'ZA12Y2C3', 'Z303XYPJ', 'S301GNF9', 'ZA12Y9PT', 'ZA180PA2', 'Z302T7ST', 'ZA11PTC3', 'ZA12FREW', 'ZA180XL7', 'Z304KR0Q', 'ZA1810CT', 'S300X3XJ', 'ZA13ZL7E', 'S301PS84', 'S300Z7AD', 'ZA10VDYA', 'S301GMR1', 'ZA16DQCD', 'S300Z4LV', 'ZA12SQA5', 'ZA13ZL67', 'ZA13R2NK', 'Z302T8WF', 'Z30544RS', 'Z304GWHB', 'Z302EPJK', 'S301GQA4', 'S301PQ1F', 'S30108L3', 'ZA189JB2', 'Z304JT1B', 'Z30417KE', 'Z30429G2', 'ZA11HYLV', 'Z304TJRT', 'ZA11J9ZY', 'Z304KCYE', 'Z302FZH2', 'Z304JWQ9', 'ZA171SJF', 'S301GVK8', 'W300THEG', 'Z304HK9B', 'Z304HZ89', 'Z300PQPQ', 'Z304JJ7R', 'ZA12PTPF', 'Z3041J67', 'Z304NGFR', '17OYTGL7T', 'Z3050DVF', 'S301GVLL', 'ZA143CPH', 'Z302T6AB', 'S301GNX4', 'ZA12FBYS', 'ZA17WKQ0', 'ZA14JB9Z', 'Z302723D', 'Z305CTEC', 'Z304KCJZ', 'Z305LRK9', 'Z305758R', 'ZA13YX8Q', 'ZA12QC8A', 'Z302T88V', 'S301K7KZ', 'ZA16DQE7', 'S301GMM5', 'Z303XYBR', 'ZA189XNN', 'Z304HTM1', 'Z302T7JZ', 'Z304L9TZ', 'Z304JPTZ', 'Z305D6RP', 'ZA12KEF8', 'Z302SYPR', 'ZA11PYLG', 'S301GLWT', 'ZA12YBEG', 'Z302T6AQ', 'Z304JKG7', 'ZA121T2A', 'Z304Y4NZ', 'Z304ZN6J', 'Z304JDL3', 'Z302SZJZ', 'Z305D4XN', 'Z305BYTD', 'ZA13RHPM', 'ZA11WJ25', 'S300WD1K', 'ZA18CEXC', 'Z304JW4P', 'S300Z7G1', 'ZA140NLE', 'ZA127XZH', 'ZA154SRE', 'ZA127Y4F', 'S301KWTE', 'Z304KCFR', 'ZA13F7AV', 'ZA16DWMX', 'Z302SQTF', 'S300WCSR', 'Z304QHY7', 'Z305A432', 'S300YSEP', 'Z305D2MG', 'Z305BE77', 'S301PCJF', 'ZA18AAH9', 'Z304JNC9', 'Z302T6J5', 'Z304176Y', 'Z302SXK7', 'ZA144P25', 'Z305D489', 'S300TS9Z', 'Z304LA3D', 'ZA13YX9S', 'Z304JLPT', 'S3010LCN', 'Z302T7P7', 'Z302SZ70', 'Z305C0ZN', 'ZA12RB2G', 'Z3025LRK', 'ZA180ZE5', 'Z302T88T', 'Z304JGD1', 'ZA18BLJ1', 'S301GRVB', 'Z304JWPZ', 'Z304HM08', 'ZA189JXJ', 'Z305D6QR', 'Z304HKAG', 'Z304KBQR', 'Z305D4WT', 'ZA16DDRB', 'S301GQJG', 'ZA111TMJ', 'ZA171V9Z', 'Z302SRFQ', 'ZA16SQ5J', 'Z302SZ1N', 'Z305DHV4', 'Z303YVGQ', 'ZA18A89Q', 'Z302F0C0', 'ZA18BTXY', 'Z30468K9', 'Z305CJQC', 'S300YC5D', 'S300Z6HR', 'S301K6XV', 'Z3059JXC', 'Z304HTKM', 'Z304LK31', 'ZA1261TH', 'ZA11HYL7', 'Z3029ZS5', 'Z305DF66', 'Z304JPN3', 'S301GTX8', 'Z305B6L0', 'Z302SZ0A', 'ZA12KYZZ', 'Z304TPPK', 'Z302T7TL', 'ZA13QBLH', 'Z305PV18', 'Z304LA61', 'ZA180RJZ', 'PL2331LAGMTS1J', 'Z305D2NT', 'S301NXNV', 'ZA13R3XX', '96ICTBPGT', 'Z304RLGF', 'Z305D2ZG', 'S301GS1L', 'Z304GWJA', 'ZA157051', 'ZA1819N0', 'ZA12TB4Q', 'S301K0WJ'}\n"
     ]
    }
   ],
   "source": [
    "set3 = set([i[1] for i in results[results[1] == 3][0].values]) \n",
    "print(len(set3))\n",
    "print(str(set3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1415\n",
      "{'ZCH0657H', 'ZJV02D6G', 'ZCH0A4CJ', 'ZCH0CZ0N', 'ZCH07W10', 'ZJV1HQ5A', 'ZCH06LCZ', 'ZCH07F4G', 'ZCH06YMP', 'ZJV0Q89C', 'ZCH0CZ5N', 'ZCH0BMET', 'ZCH0707D', 'ZJV00DR7', 'ZCH07WA6', 'ZJV03PWF', 'ZJV04NSJ', 'ZCH06K6S', 'ZCH06CHZ', 'ZCH0CDDA', 'ZCH06HQV', 'PL2331LAHDGEHJ', 'ZJV0WBSX', 'ZCH0CM1R', 'ZJV00DM8', 'ZJV00MAJ', 'ZA1805MK', 'ZA15GXGZ', 'ZCH07RTZ', 'ZCH07048', 'ZCH0CJAT', 'ZCH065XF', 'ZCH0790D', 'Z304NLHM', '17QEC1F2T', 'ZJV03EXA', 'ZCH06HFX', 'ZA180RDJ', 'ZJV1VM9J', 'ZJV2ENN5', 'ZCH0A25W', 'ZCH0BWPG', 'ZJV5MSCS', 'ZA1814FD', 'ZJV0T5EY', 'ZJV02J5W', 'ZCH06DPJ', 'ZCH0CEYE', 'Z304TK95', 'ZCH06TF7', 'ZCH07NHD', 'ZJV00A05', 'ZA1814FJ', 'ZCH0G1EJ', 'ZA1CGRE4', 'ZCH0ADQ4', 'ZCH07J8F', 'ZJV00FCE', 'ZCH0CZTV', 'ZJV0WGTV', 'ZCH0CJS9', 'PL1331LAHBXP9H', 'ZCH07W72', 'ZCH0CL8Y', 'Z3025HSR', 'ZJV1CS48', '96ICTBPMT', 'PL2331LAH45TWJ', 'ZCH0BMQB', 'ZCH06LH5', 'ZA189S3R', 'ZJV03MP4', 'ZCH06M6B', 'ZJV0WFQ2', 'ZCH074N6', 'ZCH07344', 'ZJV0WBZB', 'ZJV2ESYX', 'ZCH09VG9', '57QEWPXBT', 'ZCH0A1YD', 'ZJV4LDBY', 'ZCH06EPH', 'PL1331LAHESMAH', 'ZJV4BGZA', 'PL2331LAHDWB6J', 'ZCH07W81', 'ZJV00EXD', 'ZCH07B7C', 'PL1331LAHBSS5H', 'PL1331LAHD5BWH', 'ZCH07C5P', 'ZCH07DZ7', '57QEWPX9T', 'ZCH06J49', 'ZJV03Y00', 'PL1331LAHEPX4H', 'ZJV049MN', 'ZCH065QT', 'ZCH0ABWV', 'ZJV03G4Z', 'ZJV1HMFC', 'PL1331LAHGBUHH', 'ZJV03GLF', 'ZCH06M8Q', 'ZCH03RP8', 'ZCH0DZTY', 'ZCH0995N', 'ZA12YJK7', 'ZCH0DFXH', 'ZJV03NPS', 'ZA180QC8', '87DFWOG3T', 'ZCH0B9CL', 'ZA12Q5TH', 'ZCH07ZYJ', 'ZCH072KS', 'ZA1814JM', 'ZCH0807E', 'ZJV03DCL', 'ZJV0TCEH', 'ZJV0WFEW', 'ZCH07C5S', 'ZJV0D869', 'ZCH070VR', 'ZA180YMR', 'ZA17L2JA', 'ZCH06694', 'ZJV05KLD', '96HCTBIJT', 'ZJV000J1', 'ZCH0CE6Q', 'ZA18147N', 'ZCH086EZ', 'ZJV19NFN', 'ZCH0AD9K', 'ZCH080QC', 'ZJV2E9J0', 'PL1331LAHBYGHH', 'ZCH080JM', 'ZCH07Q3L', 'ZCH0841W', 'ZA180YM3', 'PL2331LAHBARNJ', 'ZJV00KFP', 'ZCH0CEL7', 'ZCH0AH5N', 'ZA180R52', 'ZJV065MZ', '17OYTGL3T', '57GGPDAHT', 'ZJV03PLG', 'ZCH07Q4R', 'ZJV1J8AX', 'ZCH080AD', 'ZJV00FF1', 'ZCH0CCAN', 'ZCH0EBX1', 'PL2331LAHDGEBJ', 'ZCH0CH6J', 'ZCH0B981', 'ZJV03T4K', '66GHS5MZS', 'ZJV006K3', 'ZJV03ZVD', 'ZA18CEVN', 'ZCH0A7G6', 'ZJV2FQYA', '57GGPD9NT', 'ZCH06J1G', 'ZCH0C5PL', 'ZCH07WQZ', 'ZCH0D1PC', '87DFWOGVT', 'PL1331LAHEZ7JH', 'ZCH06V8Z', 'ZCH07HYF', 'ZA17ZPZW', 'ZCH06KD2', 'ZCH09AY9', 'ZCH065CE', 'ZA180YWB', 'ZA1818AM', 'ZJV2T6YL', 'ZCH070GV', 'ZCH0818W', 'ZCH02V33', 'ZCH0CZ5L', 'ZCH0BKS2', 'ZJV030L5', 'ZJV1CS80', 'ZJV5KQY4', 'ZA180YSA', 'ZA172BWD', 'ZJV2E1CT', 'ZCH0DG4V', 'ZA1814L2', 'ZJV1KYQZ', 'ZCH06EQF', 'ZCH09VBD', 'ZCH06N6D', 'ZJV2EF1S', 'ZJV02CR4', 'ZCH0AEDG', '17OYTGKCT', 'ZJV2FRAC', 'ZJV038FA', 'ZJV136R0', 'ZCH0ADEM', 'ZA180ZVY', 'ZCH086QK', 'ZJV1C4GF', 'ZCH0D2Q6', 'ZJV00C88', 'ZCH07ZZC', 'ZJV00CJB', 'ZCH0BJ46', 'ZCH073TG', 'ZCH04Y88', 'ZA171R4R', 'ZCH07NG3', 'ZCH06FLV', 'ZJV0WGMK', 'ZCH06KQY', 'ZJV03TWR', 'ZCH0BD7Q', 'ZJV2ESCV', 'ZJV2FRSA', 'ZJV04F4Q', 'ZJV0EB23', 'ZCH09VR2', 'ZA1818F7', 'ZJV0079J', 'ZCH075WD', 'ZA10BYCZ', 'ZCH080S1', 'ZJV005EG', 'ZCH06F6S', 'ZCH0CEQ1', 'ZCH07ZDP', 'ZCH0CY39', 'ZCH0G7P9', 'ZJV1CSQ0', 'ZCH09200', '57QEWPXET', 'ZCH06ME4', 'ZA17ZPGD', 'ZJV03KCP', 'ZJV00P3B', 'ZCH0BA12', 'ZCH07SS5', 'ZCH06DJN', 'ZCH07H06', 'ZCH0BSBX', 'ZJV0TC7R', 'ZA153THS', 'ZCH0BW17', 'ZCH07NQV', 'ZCH0737J', 'ZCH0CY78', 'ZJV0WGRH', 'ZCH0CZWX', 'Z305BFRR', 'ZCH07NGJ', 'ZJV00NYA', 'ZJV2E81N', 'ZJV059JS', 'ZJV12DEB', 'ZCH0D49Y', 'ZCH0BXAL', 'ZA10ATFH', 'ZJV0TCAR', 'ZCH084S5', 'ZA1819BW', '57GGPDADT', 'ZJV2EFKN', 'PL2331LAHDRWJJ', 'PL1331LAHD6WVH', 'ZJV00C8N', 'W301AY5V', 'ZCH05KWT', 'ZCH095SY', 'ZCH0AJFN', 'ZCH0CALM', 'ZCH07XM9', 'ZJV00HVH', 'ZCH07H2T', '57HBWEF2T', 'ZJV2E0K4', 'ZCH084BR', 'ZJV004VF', 'ZCH0CZ94', 'ZJV03N3H', 'ZJV03NCN', 'ZA181JB7', 'ZA13FB8F', 'ZCH07QCM', 'ZCH07BAE', 'ZCH0931H', 'ZJV12R9J', 'ZA1805QT', 'ZCH06VQE', 'ZCH0D8BF', 'ZCH06DX3', 'ZJV1CSBM', 'ZJV03DGT', 'ZCH07FDM', 'ZCH0CW8H', 'ZJV02LNB', 'ZCH0B8ZF', 'ZCH0761G', 'ZJV00DCH', '57QEWPXCT', 'ZA18148A', '96HCTBI3T', 'ZJV00917', 'ZCH0B0DB', 'ZJV03HNX', 'ZCH0C1R6', 'ZCH0DT14', 'ZCH06HFY', 'ZCH068EH', 'ZCH0BN33', 'ZJV00CJA', 'ZCH07B55', 'ZCH09WZZ', 'ZCH06E1J', 'ZA11JNJT', 'ZJV0ZW66', 'ZCH0DFNT', 'ZJV12R62', 'ZJV2FPYG', 'ZA11QHJ1', 'ZJV0JK88', 'ZCH06F9N', 'ZCH07W16', 'ZCH04TKE', 'ZCH0E54J', 'ZCH084E3', 'ZA1818RN', 'ZCH0BJXP', 'ZJV1F9AQ', 'ZJV1CGE6', '57GGPD93T', 'ZCH0AHFA', 'ZCH086TL', 'ZCH0681Z', 'PL2331LAHDTJHJ', 'PL2331LAH9R55J', 'ZCH0704J', 'ZCH06K3T', 'ZCH08PZD', 'ZCH0AJ7S', 'ZJV2FP0B', 'ZCH0CTQ2', 'ZJV0WCQN', 'ZCH0FKZN', 'ZCH081P8', 'ZJV00X8A', 'ZJV2E6AS', 'ZCH0D3JH', 'ZCH08YC3', 'ZA180E3W', 'ZCH07JDQ', 'ZJV0QNMW', 'ZCH08NEF', 'ZCH0CZQ1', 'ZCH07HKG', 'ZCH0CBZZ', 'ZCH09H7D', 'ZCH073D9', 'ZJV2SSR4', 'ZCH06MPR', '17OYTGKOT', 'ZCH0B3Z2', 'PL1331LAHD5RRH', 'ZCH0D2AJ', 'ZCH0C5JJ', 'ZCH074AW', 'ZCH09477', 'ZCH0C00R', 'ZCH0AZQV', 'ZA189XD8', 'ZCH0C9N2', '96ICTBQTT', '57QEWPXTT', 'ZJV0Q3WC', 'ZA16YQXF', '96HCTBHRT', 'ZJV03KM5', 'ZA18145R', 'ZCH0A1K5', '96ICTBQ4T', 'ZJV0XZ34', 'ZJV017JS', 'ZCH04ZN1', 'ZCH06ZQA', 'ZA17G86N', 'ZCH0AJT3', 'ZJV0DH3Q', 'ZA1890DY', 'ZCH0BVVC', 'ZA17VHYX', 'ZCH0G2VB', 'ZJV05X0J', 'ZJV0WEQM', 'ZCH06LPP', 'ZA12T10W', 'ZJV0WGMA', 'ZCH07Y48', 'ZA17ZEY3', 'ZJV2QMLV', 'ZJV1ZG46', 'ZJV000NM', 'ZCH0DVJN', 'ZJV007SV', 'ZCH044F4', 'ZCH0BJ05', 'ZCH06L5P', 'ZCH0B72W', 'PL1331LAHDXUUH', 'ZCH0CSVW', 'ZCH07W6X', 'ZA174K5F', 'ZCH01VVW', 'ZCH07418', 'ZCH09V7R', 'ZJV00CKB', 'ZCH08B81', 'ZJV0TC00', 'ZJV0WCS2', 'ZCH07Q96', 'PL1331LAHD3PTH', 'ZCH07CH1', 'ZA1814SY', 'ZCH07BEE', 'ZJV0M2TR', 'ZCH0508S', 'ZCH07HX7', 'ZCH0CFH7', 'ZCH07HWT', 'ZJV1C4BH', 'ZJV00FVH', 'ZCH0EQAJ', 'ZCH0840Z', 'ZCH0926P', 'ZA11TXM3', 'ZCH07ZRN', 'ZJV03P7E', 'ZCH07ZW4', 'ZCH0CE31', 'ZCH092EC', 'ZJV04HXS', 'ZCH0DG22', 'ZA1810B6', 'ZJV2E903', 'ZA1816BZ', 'ZCH0B9AY', 'ZJV0WFE6', 'ZCH0D21B', 'ZCH07910', 'ZCH07P9Q', 'ZCH07JA4', 'ZCH068BL', 'ZCH07XHS', 'ZCH0B5VZ', 'ZJV0HAXB', 'ZCH0D2GM', 'ZJV0WEYV', 'ZCH07E56', 'PL2331LAHAWSPJ', 'ZCH0CYXM', 'ZA17ZF93', 'ZCH07X43', 'ZCH0AQ1Q', 'ZCH07YKN', 'ZJV03PP2', 'ZJV121PT', 'ZJV0WG58', 'ZJV0WCVY', 'ZJV00HT3', 'ZCH07SM7', 'ZJV5F4LW', 'ZCH0CF3L', 'ZCH0CHWK', 'ZCH08XHJ', 'ZCH0CZEK', 'ZCH05FM3', 'ZCH07JBL', 'ZCH06HT6', 'ZJV2EB2W', 'ZA180Z3W', 'ZCH07C1S', 'ZA1819DM', 'ZJV00D6N', 'ZJV2E9J2', 'ZCH0BPTZ', 'ZJV03G3A', 'ZCH04X6P', 'ZCH09GZ3', 'ZJV05S8C', 'ZCH06VVP', 'ZJV2ECQW', 'ZJV03JDV', 'ZJV04KE5', 'PL1331LAHENJ2H', 'ZA131FYF', 'ZJV058H7', 'ZCH07VX2', 'ZCH0B7KM', 'ZJV008TM', 'ZCH092S4', 'ZJV0CF01', 'ZA16CLZT', 'ZJV03SVX', 'ZA180YYT', 'ZA180R3Y', 'ZA173SYB', 'ZJV0KB9Z', 'ZA180YSX', 'ZA12PTFW', 'ZCH0B663', 'ZCH0CLCF', 'ZJV10G9G', '564ESIZHS', 'ZCH0CCB4', 'ZCH0CZBK', 'ZCH086XE', 'ZJV2F3D5', 'ZCH0CXS2', 'ZCH095CR', 'ZCH072H4', 'ZCH0CY6W', 'ZCH0BP0P', 'ZCH0A2DH', 'ZJV00F8H', 'ZCH06PC6', 'ZCH0BR0M', 'ZCH0CAKT', 'ZCH09LGZ', 'ZJV0NQFE', 'ZJV01J6F', 'ZCH0D1FB', 'ZA10JGLG', 'ZCH0CCQ7', 'ZCH07C9X', 'ZCH0CE5D', 'ZCH07HBQ', 'ZJV00EGC', 'ZCH0AMQN', 'ZJV1CG3X', 'ZCH06DXS', 'ZCH07DXQ', 'ZCH02LQX', 'ZCH0DGNC', 'ZCH0BGFH', 'ZCH07C0Y', 'ZCH08AC3', 'ZJV00DV2', 'ZJV3YXCB', 'ZA12XGTD', 'ZA181ABA', 'ZCH076TV', 'ZJV05B4G', 'ZCH06X1E', 'ZCH06FBG', 'ZCH07W42', 'ZCH0B172', 'ZJV2EFFS', 'ZCH0CE3Z', 'ZJV02TBM', 'PL2331LAHDE5NJ', 'ZA1805V0', 'ZCH01EXE', 'PL1331LAHD29UH', 'ZJV00DK9', 'ZCH07M4C', 'ZA12Q5X7', 'ZCH0CYQR', 'ZCH06VE2', 'ZCH06WRF', 'ZJV5HESD', 'ZCH0DYK0', 'ZCH06F1A', 'ZCH06KL4', 'ZCH081EP', 'ZCH06VBD', 'ZCH08B2G', 'ZCH0BXK2', 'ZA189Y1M', '96HCTBH7T', 'ZCH06GPN', 'ZCH07B71', 'ZJV5HVRQ', 'ZCH091PJ', 'ZCH0CJT2', 'ZJV16128', 'ZCH06HQK', 'ZJV0SWXA', 'ZCH0547F', 'ZCH07NMZ', 'ZJV1C4CN', 'ZCH06VQC', 'ZJV1CGK8', 'ZCH0B34Z', 'ZCH0838E', 'ZCH0E0AV', 'ZCH0968V', 'ZCH07HKF', 'ZCH0AWQY', 'ZA153S9S', 'ZJV2DZE3', 'ZCH09JYH', 'ZCH08A2N', 'ZCH0CHNQ', 'ZCH0D1BJ', 'ZCH0C08W', 'ZCH06FJN', 'ZJV00FBJ', 'ZCH0817E', 'ZCH077D1', 'ZJV03NB0', 'ZCH07M07', '96ICTBPTT', 'ZJV2EE4A', 'ZA17ZEZE', 'ZCH09645', 'ZCH06H6M', 'ZCH0BABW', 'ZCH0D2GK', 'ZCH07RK0', 'ZCH0705J', 'ZJV0WEHX', 'ZJV0WG22', 'ZJV007FG', 'ZJV065T5', 'ZJV00FQZ', 'ZCH07Z1F', 'ZCH05BD5', 'ZJV0PRLM', 'ZA15AX9X', 'ZCH09B79', 'ZJV00CDN', 'ZJV0WFMV', 'PL2331LAHAT7VJ', 'ZCH03BMN', 'ZCH0BTEJ', 'ZCH0CY4F', 'ZJV1CSTH', 'ZCH06L88', 'ZCH05L1Z', 'ZJV025FK', 'ZCH07STY', 'ZCH07JLH', 'ZCH0CXLV', 'ZCH080MT', 'ZA1818CA', 'ZCH06E9G', 'ZA17ZPT9', 'ZCH08BQ9', 'ZCH07F30', '564ESIYPS', 'ZJV001EW', 'ZJV17VH7', 'ZJV00HTE', 'ZJV009V2', 'ZCH06N24', 'ZJV030RR', 'ZCH0CZTQ', 'ZCH083PD', 'ZJV00PQ3', 'ZJV2EGTN', 'ZCH07S1G', 'ZJV03N75', 'ZJV01Y53', 'ZJV0ZECX', 'ZCH06LGG', 'ZCH07B3R', 'ZJV2EECT', 'ZCH072BF', 'ZCH075V0', 'ZJV5J8F1', 'ZCH06FHZ', 'ZCH07E4G', 'ZCH09753', 'ZCH06TNZ', 'ZCH06MRJ', 'ZCH07T76', 'ZCH0C891', 'ZCH06HD5', 'ZCH06XAA', 'ZCH07G7A', 'ZCH06MSQ', 'ZA1819HK', 'ZCH0CY2A', 'ZCH09H5J', 'ZCH0920F', 'ZCH086VV', 'ZA10NEVW', 'ZCH0BXQS', 'ZCH09V58', 'ZJV0Y5PL', 'ZA18DFFK', '563ESIX1S', '57GGPD9DT', 'ZJV3BXAK', 'ZCH07ERS', 'ZCH0CK24', 'ZCH07WBW', 'ZCH0AE85', 'ZCH09DGN', 'ZCH06MQH', 'ZCH0BJM3', 'ZJV006P9', 'ZA180RE9', 'ZCH0830T', 'ZJV03KEK', 'ZJV0WEKK', 'ZCH0973A', 'ZCH0BSMP', 'ZA170TQ1', 'PL1331LAHD69MH', 'ZJV1377T', 'ZCH0CFP5', 'ZJV00EAZ', 'ZCH06WWK', 'ZA1810C1', 'ZJV2DY9K', 'ZJV1C4BB', '96ICTBQ3T', 'ZCH0DY2X', 'ZA153STG', 'ZCH07ZYC', 'ZA16CR18', 'ZCH072D6', 'ZCH0B7RN', 'ZCH0CZV4', 'ZCH0DXLH', 'ZCH0CDHE', 'ZJV00D2K', 'ZCH07VR2', 'ZJV00N9A', 'ZCH06DMQ', 'ZA153S86', 'ZJV4JLHP', 'ZJV00FFN', 'ZCH07GDG', 'PL1331LAHD3T1H', 'ZCH07WH8', 'ZCH0CE37', 'ZJV117ZK', 'ZJV2FRKH', 'ZCH0BPA8', 'ZCH067ZH', '57QEWPXNT', '57QEWPXKT', 'ZCH089NB', 'ZJV0F8M6', '57HDTBTYT', 'ZCH07ZDK', 'ZJV03V0A', 'ZJV0P7Y0', 'ZCH0D3BM', 'ZCH0CZBH', 'Z305GV90', 'ZJV5JSG5', 'ZCH0DL9T', 'ZA1818R3', 'ZCH07XL7', 'ZCH0APCY', 'ZCH081FV', 'ZA174KNX', 'ZJV03KMS', 'ZA173EB5', 'ZJV00DDS', 'ZCH06D55', 'Z305DXS7', 'ZJV000NQ', 'ZCH06GX4', 'ZCH0BML8', 'ZJV1H4BZ', 'ZCH0AL23', 'ZCH076SX', 'ZCH09MYS', 'ZA12RAZA', 'PL1331LAHBGMNH', 'ZA17ZNZJ', '87DFWOFRT', 'ZCH0D3KZ', 'ZCH07WYA', 'ZJV03N38', 'ZJV1XE1D', 'ZCH0830F', 'ZCH0C6NY', 'ZJV5ETX7', 'ZCH0A2Y7', 'ZCH06J46', 'ZCH08N8S', 'ZJV021L1', 'ZJV0504G', 'ZA17ZNQ9', 'ZCH06J61', 'ZCH086ZH', 'ZJV04Q83', 'ZCH0784T', 'ZJV03W5T', 'ZCH06E6B', 'ZA10YPL3', '96HCTBHWT', 'ZCH07RN1', 'ZA180XD5', 'ZCH07E70', 'ZCH038Z5', 'ZCH0CDBX', 'ZJV00FJ4', 'ZCH07VQ5', '5732PE2HT', 'ZA18CDZ5', 'ZJV2ECXH', 'ZCH06HG5', 'ZCH0BWRA', 'ZA17347A', 'ZJV2ENS7', 'ZA180QJ5', 'ZCH0CZ7F', 'ZCH0ABTP', 'ZJV0WF02', 'ZCH0BVLB', 'ZA153S5L', 'ZCH0BJCG', 'ZCH080VT', 'ZCH069AN', 'ZCH09257', '57QEWPXST', 'ZCH07VW8', 'ZJV114F4', 'ZJV0652N', 'PL2331LAHDMBHJ', 'ZA12G1N8', 'ZJV0TQS8', 'ZCH08XHP', 'ZA180N6K', 'ZCH07BXA', 'ZCH0CJ2Z', 'ZCH07SN7', 'ZCH0D1SB', 'ZCH0BNS3', 'ZJV2E7LD', 'ZCH0FKWF', 'ZCH0B3M5', 'ZCH0E16J', 'ZA10A9SB', 'ZJV0D8LY', 'ZCH0CJ8F', '57QEWPXQT', 'ZCH0AMP6', 'ZCH07HHL', 'ZA1819EY', 'ZCH0APMZ', 'ZCH07AYE', 'ZCH07GE0', 'ZJV0Z3H0', 'ZCH084A7', 'ZCH070B8', 'ZCH0CC3Q', 'ZCH09Z0A', 'ZJV0WF83', 'ZCH0C6VX', 'ZCH06LDL', 'ZCH09MGG', 'ZCH07JQX', 'ZCH08BPM', 'ZCH095YG', 'ZCH03YFN', 'ZA1814EZ', 'ZCH0DG4X', 'ZCH0B9RA', 'ZCH07BCB', 'ZCH06WP8', 'ZCH044NK', 'ZCH06ZR3', 'ZCH06H58', '57RFWNHFT', 'ZJV03MZ2', 'ZCH0C1SC', 'ZCH086DZ', 'ZCH09KDF', 'ZCH0CZ0K', 'ZCH060TB', 'ZJV0D8N8', 'ZJV2EB4D', 'ZJV03N8M', 'ZCH06GQE', 'ZJV10J65', 'ZCH065B6', 'ZCH0D2GD', 'ZCH0C20V', 'PL2331LAHBB6AJ', 'ZJV0WFST', 'ZCH07J18', 'ZCH07VQ8', 'ZCH0B4YA', 'ZCH0ADRN', 'ZCH089LH', 'ZCH095WZ', 'ZJV2E12G', 'ZA18198G', 'ZA180R77', 'ZCH085F2', 'ZCH074DQ', 'ZCH037K5', 'ZJV00W03', 'ZJV03VVA', 'ZJV03JH3', 'ZJV02PBH', 'ZCH080T7', 'ZCH0BFCV', 'ZCH08B0N', 'ZJV03PRG', 'PL2331LAHAWS6J', 'ZCH0DG3H', 'ZCH0CJD5', 'ZCH0997R', 'PL1331LAHBRPHH', 'Z305D34Z', 'ZCH07ZTY', 'ZCH06EE3', 'ZCH06FFA', 'ZJV2E7H6', 'ZA133J4B', 'ZCH0BWRT', 'ZCH07Q93', 'ZJV5E2M1', 'PL2331LAHDUMVJ', 'ZJV110PW', 'ZCH092TK', 'PL2331LAHDXA8J', 'ZCH0BC0N', 'ZCH06LMM', 'ZCH0FKSQ', 'ZJV0RBAK', 'ZCH06WEZ', 'ZJV0RP90', 'ZJV2E87R', 'ZCH07X9G', 'ZCH07NZS', 'ZCH0BHTZ', 'ZJV001HR', 'ZCH080T0', '57HBWEEST', 'ZCH0BTEP', 'ZCH06L67', 'ZCH06J2N', 'Z305DH9R', 'ZJV0VSDQ', 'ZCH084HD', 'ZCH07Q8A', 'ZJV03P1E', 'ZCH097GA', '96ICTBQ2T', 'ZCH0CZRM', 'ZJV1CGL8', 'ZCH06WSZ', 'ZCH0CJPF', 'ZCH0ADNJ', 'ZCH06EJ0', 'ZCH06TH0', 'ZA108H7W', '17OYTGLGT', 'ZCH0CA7Z', 'ZCH0A87H', 'ZJV0WFRQ', 'ZJV2E8Q9', 'ZCH07BDG', 'ZCH0E9FM', 'ZCH07CHP', 'ZCH012JE', 'ZJV0WGT8', 'ZCH07VHQ', 'ZJV0G9YJ', 'ZJV0YPQE', 'ZJV1K5HB', 'ZCH037WQ', 'ZCH02Z5J', 'ZJV106AL', 'ZCH06G1T', 'ZJV1YGCP', 'ZCH06NDW', 'ZA153SKC', 'ZJV00EY2', 'ZCH0CZC6', 'ZCH09D6M', 'ZJV03NQE', 'ZCH0AB9N', '87DFWOFZT', 'PL2331LAHDBGPJ', 'ZCH06HWP', 'ZJV00CDQ', 'ZJV03W1L', 'ZCH072V4', 'ZJV03N4N', '5641SFRAS', 'ZCH0CZ29', 'ZCH0ACRA', 'ZCH0B4LQ', 'ZJV4VYVX', 'ZCH0CJLL', 'ZA108HT7', '96HCTBHXT', 'PL2331LAHDS02J', 'ZA10MCJ5', 'ZCH07GTF', 'ZCH09ALF', '66HLS2P2S', 'ZCH0CHT6', 'ZCH07CRX', '96ICTBPZT', 'ZJV03DB3', 'ZCH07EHF', 'ZCH0CJ8H', 'ZA18189Z', 'ZJV2EN41', 'ZJV2EB6S', 'ZJV005A9', 'Z304DWCH', 'ZCH0A2M8', 'ZCH0DWXR', 'ZJV04J56', 'ZCH09SPP', 'ZCH08YRL', 'ZCH07H70', 'ZJV0YP2X', 'ZA180YGA', 'ZCH07816', 'ZA17ZF5W', 'ZJV5MFL7', 'ZJV00BLV', 'ZJV00HBT', 'ZCH0FL2A', 'ZA143SAS', 'ZA171S1E', 'ZJV03NQB', 'ZJV2GCDD', 'ZCH0B7FX', 'ZCH0931T', 'ZCH0BTAS', 'ZCH0CZCN', 'ZJV0KB4L', 'ZCH077J5', 'ZCH078YM', 'ZCH0B89N', 'PL1331LAGK9JRH', 'ZCH0CZNA', 'ZJV0WFYF', 'ZCH0CZXB', 'ZJV5MDP1', 'ZJV05LJM', 'ZA17ZNGC', 'PL2331LAH3109J', 'ZCH06MNK', 'ZJV03NK5', 'ZJV03VYV', 'ZJV0016J', 'ZJV0T50C', 'ZA130MZT', 'ZCH06VN1', 'ZCH06LY0', 'ZA1815AA', 'ZJV0XPN5', 'ZJV000NZ', 'ZJV5LWDJ', 'ZCH084F7', 'ZCH089VQ', 'ZCH06H3V', 'ZCH0BK7Z', 'ZJV2ECQC', 'ZCH06W1Q', 'PL2331LAHB92ZJ', 'ZCH0ASMB', 'ZCH0D3LA', 'ZCH078KD', '96HCTBI9T', 'PL1331LAHBS2EH', 'ZJV00BMQ', 'Z305DHXM', 'ZJV2EQCS', 'ZA17ZDJX', 'ZA180XCQ', 'ZCH07Q2H', 'ZA15531N', 'ZCH0AY0D', 'ZCH0CZ28', 'ZJV00C4T', 'ZCH06MF7', 'ZCH0AEGA', 'ZCH0923X', 'ZJV1CSEE', 'ZJV0XZ8A', 'ZJV03JEL', 'Z304LDL0', 'ZCH07415', 'ZJV0BMW1', 'ZCH07X34', 'ZA180Q95', 'ZCH07NZM', 'ZCH07665', 'ZJV2S69B', 'ZCH07BBG', 'ZCH0B4F5', 'ZJV2EN1K', 'ZJV00C9E', 'ZCH066VK', 'ZJV2EH7X', 'ZCH09SL7', 'ZCH09MHJ', 'ZJV007PR', 'ZCH0BWVW', 'ZCH07XQ4', 'ZA1815CD', 'ZCH0ABNA', 'ZJV03PP4', 'ZCH07SPM', 'ZA12X0AP', 'PL1331LAHESH5H', 'ZJV10J7D', 'ZJV03DF9', 'ZCH0B0SJ', 'ZCH066YX', 'ZCH0BSMA', 'ZCH0CN2T', 'ZCH06147', 'ZCH06F9T', 'ZCH071MJ', '5641SFR8S', 'ZCH0A88N', 'ZCH07XDD', 'ZCH0A3FM', '66GHS5N1S', 'PL2331LAHDBV2J', 'ZCH0CLDX', 'ZJV1C49G', 'ZJV1C4DQ', 'PL2331LAHD87SJ', 'ZJV2E95R', 'ZA180DAX', 'ZCH0AQ0J', 'ZJV0TCH2', '5732PE2FT', '57GGPD9TT', 'ZJV2E0MT', 'ZCH0AD8M', 'ZCH070E1', 'ZCH083PT', 'ZCH07ZKV', 'ZCH07EH9', 'ZCH06WCG', 'ZCH07RV5', 'ZCH0D6ZB', 'ZA1819G2', 'ZJV2E1QW', 'ZJV03N3F', 'ZCH06EEA', 'ZCH0BT7K', 'ZJV113ZA', '5641SFRHS', 'ZCH07TG7', 'ZJV0H9QF', 'ZCH0CJC3', 'ZCH0CYTR', '96HCTBIDT', 'ZJV0WWCC', 'ZCH0818B', 'ZCH0CZDB', 'ZCH07VXA', 'ZCH072RQ', 'ZCH0GGTT', 'ZJV1NJ97', 'PL2331LAHDTTRJ', 'ZCH0EB3K', 'ZCH06HMJ', 'ZJV01VBF', 'ZCH07RYZ', 'ZJV12RY2', 'ZCH0B2VP', 'ZCH0CCKJ', 'ZCH0C5J0', 'ZCH0793L', 'ZCH0ABTR', 'ZCH0783C', 'ZA109X50', 'ZJV0WGCH', 'ZA180RAR', 'ZCH0AXMF', 'ZJV02SQB', 'ZCH0FEAV', 'ZCH07BZB', 'ZCH0DG1T', 'ZA17ZNJY', 'ZCH06PCK', 'ZCH06V2F', 'ZCH07T7C', 'ZCH0472C', 'ZJV04TMW', 'ZCH06VKQ', 'ZCH06ET2', 'ZCH0666S', 'ZCH08406', 'ZCH0D1F9', 'ZCH07TGD', 'ZA173E77', 'ZJV2EA6T', '87DFWOGDT', 'ZA180R72', 'ZCH074EG', 'ZA18CD35', 'ZJV005M5', 'ZJV03F45', 'ZCH0761Z', 'Z303QAX3', 'ZCH0966Y', '57GGPD9FT', 'ZCH074M0', 'ZJV00C1D', 'ZCH0BTJN', 'ZA10NFKE', '96ICTBPUT', 'ZCH084QG', 'ZCH06HZL', 'ZA1810KV', 'ZA143SM3', 'ZJV2EHF1', '564ESIYGS', 'ZCH0D2S8', 'ZJV1CS29', 'ZCH099N2', 'ZCH077N9', 'PL1331LAHBSTTH', 'ZCH07VWP', 'ZCH07X3Q', 'ZJV00EPV', 'ZCH0B8BW', 'ZJV2ESBP', 'ZA11VCRQ', 'ZJV38HL6', 'ZCH068XD', 'ZJV0WBV6', '87DFWOFYT', 'ZJV03N18', 'ZJV1L5N1', 'ZCH072EB', 'ZCH06LSB', 'ZCH03Y6Y', 'ZJV2E3Z6', 'ZJV2KEKY', 'ZCH07H83', 'ZCH0D3KH', 'ZCH06J2M', '96ICTBPQT', 'ZCH09WGS', 'ZCH080R7', 'PL1331LAHBJ3PH', 'ZCH09PKH', '96HCTBI6T', 'ZJV2E65H', 'ZCH07RJM', 'ZJV03PTB', 'ZCH07HK7', 'ZA13YAZR', 'ZCH0CZ97', 'ZCH0BMML', 'ZCH08A3Z', 'ZCH07X3B', 'ZA18183R', 'ZA174KE4', 'ZJV2EG1W', 'ZCH07BWD', '57QEWPXLT', 'ZCH0A24Q', 'ZCH0BMMQ', 'ZA181K0Y', '96HCTBI7T', 'ZCH099N3', 'ZJV5CE9P', 'ZCH077G2', 'ZA153S34', 'ZA180Z59', 'ZCH0C5K1', 'PL2331LAH41JTJ', 'ZJV0084C', 'ZA153SZQ', 'ZCH09RAW', 'ZCH06KZT', 'ZJV17SG6', 'ZA15530Y', 'ZJV0WEKS', 'ZJV0WCXQ', 'ZCH0CE2C', 'ZJV2ER1G', 'ZA180R10', 'ZJV4KZ2Q', 'ZCH07B11', 'PL1331LAHBYKEH', 'ZCH0B0WQ', 'ZCH06EDW', 'ZJV198EX', 'ZJV004T6', 'ZJV03JEZ', 'ZA153SHZ', 'ZCH07B5Y', 'ZA10Z4BZ', 'ZJV2E6EY', 'ZCH0BXKL', 'ZJV0XEDZ', 'ZA1818HP', 'ZJV2S4YQ', 'ZCH0DKZR', 'ZJV01W64', 'ZCH0CYBT', 'ZCH0AZ6R', 'ZCH0CY7D', 'ZCH06DP9', 'ZCH0B8G4', 'ZCH0DXXE', 'ZJV2FQWM', 'ZA126676', 'ZCH09K02', 'ZJV0WG3L', 'PL2331LAH3WU2J', 'ZA180R1C', 'ZCH06V6C', 'ZJV0WBVF', 'ZCH0CJFK', 'ZCH0CJC5', 'ZCH07JRJ', 'ZCH07T9K', 'PL1331LAHBSPAH', 'Z303LQN7', 'ZCH0E7WL', 'ZA1818FT', '57HBWEE3T', 'ZJV12QK3', 'ZJV0T51G', 'ZJV03FVT', 'ZJV00CNM', 'ZCH0BVDJ', 'ZCH0BE2C', 'ZCH0516L', 'ZJV00F20', 'ZCH07NV4', 'ZCH0AJNX', 'ZCH07AWC', 'ZCH06X32', 'ZCH07VZ1', 'ZCH07RLC', 'ZJV5JWCJ', 'ZCH0CK5V', 'ZCH0CK4L', 'ZCH09FCW', 'ZCH09NX1', 'ZA180QAN', 'ZCH0D3A3', 'ZCH07FQP', 'PL1331LAHESMRH', 'ZCH09KAF', 'ZCH0CES4', 'ZCH06TYD', 'ZCH06LYW', 'PL1331LAHEU86H', 'ZCH01CPY', 'PL2331LAH3HXGJ', '96ICTBQET', '66GHS5MMS', 'ZCH0CZ8G', 'ZJV02XH7', 'ZCH09SVS', 'ZJV0WFMG', 'ZJV2E6LL', 'ZCH072R3', 'ZA15YBJM', 'ZA1814NS', 'ZJV0WGT2', 'ZJV2FRZE', 'ZCH06EQZ', 'ZCH05E2V', 'ZCH06MN9', 'ZJV03WYL', 'ZJV19EWJ', 'ZCH0A0DD', 'ZA16DSXP', '17OYTGLNT', '96HCTBI1T', 'PL1331LAHESH6H', 'ZCH0BW5F', 'ZCH09QZJ', 'ZA18106W', 'ZCH083YQ', 'ZCH09S7D', 'ZCH0BYK7', 'ZJV00GQH', '564ESIYTS', 'ZCH09MVP', 'ZCH089YR', 'ZA13GZCA', 'ZCH0CA2L', 'ZCH06VFA', 'ZA1810KB', 'ZCH07TJ4', 'ZCH08014', 'ZJV12HQX', 'ZA180QHT', 'ZJV2FPXG', 'ZJV0ZXGM', 'ZCH0CY92', 'ZCH0CJMV', 'ZJV2EFWC'}\n"
     ]
    }
   ],
   "source": [
    "set4 = set([i[1] for i in results[results[1] == 4][0].values]) \n",
    "print(len(set4))\n",
    "print(str(set4))"
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
 "nbformat_minor": 2
}
