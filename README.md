
#### CS 7461 Project 21: [Akarshit Wal](https://github.com/Akarshit), [Gnanaguruparan Aishvaryaadevi](https://github.com/Aishvaryaa), [Karthik Nama Anil](https://github.com/KarthikNA), [Parth Tamane](https://github.com/parthv21), [Vaishnavi Kannan](https://github.com/Vaishnavik22)  

<p align="center">
    <img src="images/coverimg.png">
</p>

# Introduction
Hard disk failures can be catastrophic in large scale data centers leading to potential loss of all data. To alleviate the
impact of such failures, companies are actively looking at ways to predict disk failures and take preemptive measures.
Traditional threshold-based algorithms were successful in predicting drive failures only 3-10% of the times[1]. Thus,
we saw a shift to learning based algorithms that use Self-Monitoring, Analysis and Reporting Technology (S.M.A.R.T)
attributes to make predictions. These attributes are different hard drive reliability indicators of an imminent failure.
In recent times, people have applied insights and learnings obtained from analysing hard drives of one vendor to other
vendor using tranfer learning techniques[2]. These models either used drives from specific vendors to achieve high
F score[2] or used a subset of data and selected attributes[3]. In this project, we have explored unsupervised and
supervised learning techniques to predict and analyze hard drive crashes. The objective of using both supervised and
unsupervised algorithms is to make a comparison between them.
If companies are able to predict failure of their hard-drives, it would save them tons of money and help them gain customer trust.
Admittedly there are cases where the disk failure cannot be predicted, like electricity failure in the server, natural hazard etc. But most of the hardware failure doesn't happen overnight. A hard-disk starts to show reduced performance in some of the attributes before failing. Recognizing these attributes and training a machine learning model to predict failure based on these attributes is the goal of the project.

# Motivation

# Problem Statement

The problem is to predict when a disk is going to fail. To form this prediction, we are given the S.M.A.R.T attributes of hark-disks along with the date they were collected on. We have fed these attributes to our machine learning algorithms and predicted the results.
We have also compared difference machine learning algorithms and their accuracy on the same dataset. This has helped us formulate which algorithms better suit the dataset and also achive better results.

# Dataset

<p align="center">
    <img src="images/backblazelogo.png">
</p>

# Methodology

<p align="center">
    <img src="images/methodology.png">
</p>


## Data Cleaning

## Feature Selection

## Unsupervised Learning

## Supervised Learning

# Results

### Random Forest

<p align="center">
    <img src="images/RF.png">
</p>


### XG Boost

<p align="center">
    <img src="images/XGB.png">
</p>

### Isolation Forest

<p align="center">
    <img src="images/IF.png">
</p>

# Conclusion

# References
1. C. Xu, G. Wang, X. Liu, D. Guo, and T. Liu. Health status assessment and failure prediction for hard drives with recurrent neural networks. IEEE Transactions on Computers, 65(11):3502–3508, Nov 2016.
2. Mirela Madalina Botezatu, Ioana Giurgiu, Jasmina Bogojeska, and Dorothea Wiesmann. Predicting disk re- placement towards reliable data centers. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.
3. Jing Shen, Jian Wan, Se-Jung Lim, and Lifeng Yu. Random-forest-based failure prediction for hard disk drives. International Journal of Distributed Sensor Networks, 14(11):1550147718806480, 2018.
4. Backblaze. Backblaze hard drive state, 2020.
5. J. Li et al. Hard drive failure prediction using classification and regression trees. In 44th Annual IEEE/IFIP International Conference on Dependable Systems and Networks, Atlanta, GA, 2014, 2014.

----

Dataset = [Hard Drive Data and Stats - Hard drive failure data and stats from 2019](https://www.kaggle.com/jackywangkaggle/hard-drive-data-and-stats)

Source = [Hard Drive Data and Stats](https://www.backblaze.com/b2/hard-drive-test-data.html)

# Smart Stats List

[List of all S.M.A.R.T. Attributes Including Vendor Specific](https://www.data-medics.com/forum/list-of-all-s-m-a-r-t-attributes-including-vendor-specific-t1476.html)

[List of Public SMART Attributes](http://www.t13.org/Documents/UploadedDocuments/docs2005/e05173r0-ACS-SMARTAttributes_List.pdf)

[Western Digital SMART attributes](https://sourceforge.net/p/smartmontools/mailman/message/23829511/)

