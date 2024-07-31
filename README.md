# Credit Scoring

## Introduction
This project focuses on developing a data modeling approach for a credit dataset to better understand customer loan profiles. We conducted a primary statistical analysis, handled missing values and outliers, and engineered relevant features. Three machine learning models were developed to predict the probability of default: Logistic Regression, Random Forest, and Deep Neural Network.

## Primary Statistical Analysis
We began with an exploratory data analysis, summarizing the dataset using descriptive statistics and visualizations. The dataset contained both categorical and numerical variables, with 310,704 observations in the training set and 20,600 in the test set. Key steps included:
- Identifying missing values and handling them appropriately.
- Detecting and addressing outliers using statistical and visual methods.

## Data Pre-processing
### Missing Values
We addressed missing values by:
- Assigning "Unemployed" for missing `emp_title`.
- Using "< 1 year" for missing `emp_length`.
- Imputing the mean for missing `dti`, `inq_last_6mths`, and `revol_util`.

### Outliers
Outliers were managed by setting thresholds for numerical variables, and observations exceeding these thresholds were excluded.

### Feature Engineering
Key steps included:
- Correlation analysis to understand relationships between variables.
- Transforming categorical features into numerical values.
- Creating new features such as employment stability and income-to-loan ratio.

## Modelling
### Feature Selection
Features were selected based on their correlation with the target variable, `risk`. Weakly correlated features and those with high dispersion were dropped.

### Logistic Regression
A logistic regression model was developed and evaluated using metrics such as accuracy, sensitivity, and specificity. The ROC curve and AUC score were also computed.

### Random Forest
A random forest model with 100 trees was developed, showing better performance in terms of accuracy and specificity compared to the logistic regression model.

### Deep Neural Network
A deep neural network was implemented but did not outperform the random forest model. The lower specificity indicated a higher rate of false positives.

## Conclusion
The random forest model emerged as the best performer, balancing accuracy and sensitivity effectively. Future work could focus on fine-tuning models and exploring ensemble techniques to enhance predictive performance.
