Detect Outliers in Clusters
===

### Prerequisites

You need to use Python 3.7 and later version and installed jupiter noteboosk.

## Install needed python modules and dependencies

```markdown
pip install --upgrade pip
pip install jupyter
jupyter notebook
http://localhost:8888
```

## distances_in_clusters.ipynb

This is a sample how we can detect outliers in each cluster.

Dataset (a CSV file) has many columns. Each column is a dimension and row is an observation, case or better say a data point in that n-dimensions' space. 
It is easy to present clusters (groups) in 2 or 3 dimensions' space but it is hard to visualize groups in 10 or 50-dimensions' space. 
For our sample, we use 2 dimensions where each group is a circle of dots on the plot. But in reality, there could be 100, 500-dimensions' space. 
Luckily, the math under the hood stays the same no matter how many dimensions we have.

1) First we cluster data into groups (clusters). Each cluster has its own center.
2) Second we calculate distances from every data point to its cluster center. And show distances in terms of STD.

On the picture, you can see two clusters and each has its own center. We label each data point with a number of STD - how far that 
point lies from the center of its cluster in terms of STD.

We calculate STD among all distances in each cluster from each data point to the center of its cluster. Then every distance is divided by its cluster's STD.

## outliers_plot_with_slider.ipynb

Select outliers dynamic demo

Below a demo for just one cluster (group) of data points. This demo shows how to select outliers by given a predefined number of outliers in term of STDs.

- Run all cells the cell below and
- Move the slider using your mouse

Moving the slider you change the number of STDs and select outliers those greater than a given number of outliers.

## gmm_probs.ipynb

Gaussian Mixture Model allows us to count on non-spherical clusters. 
If a cluster is stretched cloud of data points we can use Gaussian Mixture Model (GMM) 
to count of covariance of the data correctly.

## similarity.ipynb

Sample how to use similarity in the cluster to calculate the weighted average price 
for a particular data point.

## License

This project is licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
