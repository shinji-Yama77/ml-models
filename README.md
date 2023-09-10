# Diabetes Project Topic

Predicting the presence of diabetes (1) vs the absence of diabetes (0)


## Research Question

Logistic Regression vs Knn on predicting the presence of diabetes

**Why are we using these two models?**

- Both are suitable for binary classification


## Purpose of this project


1. Improve my knowledge of current ML supervised learning algorithms by implementing them from scratch and evaluating them using certain metrics

2. I am interested in healthcare and want to begin my ML journey with some interesting and manageable health data. 

3. Diabetes are common and can often be caused due to genetic mutations. Early detection is key so I wanted to build a model that can accurately determine the presence and absence of diabetes in a current individual. 

4. Practice for future experiementation with big data including cancer, and other types of health conditions. This includes CNN for MRI imaging, and CT imaging specifically. 

## Outline of this project

### Data Exploration

To understand the kaggle dataset we are working with, it is important to first performe some baseline exploratory analysis. To do this we must first ask ourselves a few questions:

1. **Is this dataset imbalanced?**
- We can measure this by grouping each class to find the proportion of each class relative to the number of examples. 

2. **Is this dataset normally distributed?**
- 

3. **Does multicolinearity exist?**


4. **Which features are important? Which features are good predictors for classification?**

5. 


## Logistic Regression model

**Sigmoid function**: We define the sigmoid function so that the outputs are calculated to be between 0 and 1

**Cost function**: Defined as the total average loss. Binary classification uses a cross entropy. 

**Gradient Descent**: Calculating the partial derivatives for all parameters (weights, biases) with respect to the cost function. 

**Update step**: We simulatenously update the weights and biases by multiplying the learning rate by the partial derivatives for each feature. 

__**Analysis**__: By visualizing the cost function vs iteration graph, I noticed that feature scaling was necessary as it occured to be oscillating. By utilizing the standard scaler library and normalizing the training data, the cost function started to decrease with every iteration. 

__**Peformance**__: 








### KNN Model




### Conclusions/Findings




### Evaluation Metrics

- There are multiple metrics for evaluating classification problems. However since this dataset isn't imbalanced, we will be evaluating the performance of our two algorithms by measuring their accuracy. Furthermore, a confusion matrix will be visualized so other metrics such as precision and recall can be considered later. 



### Future Consideration

- I will be deploying this model online so our algorithm will be able to determine whether you may have the presence of diabetes or not. 








