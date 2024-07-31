if (!require("psych")) install.packages("psych")
library(psych)
if (!require("dplyr")) install.packages("dplyr")
library(dplyr)
if (!require("gmodels")) install.packages("gmodels")
library(gmodels)
if (!require("tidyverse")) install.packages("tidyverse")
library(tidyverse)  
if (!require("corrplot")) install.packages("corrplot")
library(corrplot)
if (!require("caret")) install.packages("caret")
library(caret)
if (!require("pROC")) install.packages("pROC")
library(pROC)
if (!require("ROCR")) install.packages("ROCR")
library(ROCR)
if (!require("randomForest")) install.packages("randomForest")
library(randomForest)
if (!require("keras")) install.packages("keras")
library(keras)
if (!require("tensorflow")) install.packages("tensorflow")
library(tensorflow)
if (!require("lubridate")) install.packages("lubridate")
library(lubridate)


# Loading Data
# ----------------------------------------------------------------------------
train_dataset <- read.csv("/Users/riccardogurzu/Desktop/predictive analytics/train_validation_kaggle.csv", 
                          na.strings = c("", "n/a"), 
                          stringsAsFactors = T)
# Remove the 'id' column from the dataset
train_dataset <- train_dataset[, !(names(train_dataset) %in% c("id"))]

test_dataset <- read.csv("/Users/riccardogurzu/Desktop/predictive analytics/unseen_kaggle.csv", 
                         na.strings = c("", "n/a"), 
                         stringsAsFactors = T)

# View datasets
View(train_dataset)
View(test_dataset)

#Check details
summary(train_dataset)
summary(test_dataset)
head(train_dataset, n=5)
tail(train_dataset, n=5)
describe(train_dataset)
describe(test_dataset)

str(train_dataset)
str(test_dataset)


# Create a bar plot of the 'risk' to check the balance of the dataset
barplot(table(train_dataset$risk), 
        col = c("lightblue"), 
        main = "Risk", 
        ylab = "Frequency")


# ----------------------------------------------------------------------------
#Data #Pre-Processing
# ----------------------------------------------------------------------------
##Data-Type
str(train_dataset)
str(test_dataset)

# ----------------------------------------------------------------------------
##Missing Values

###Train Dataset
sapply(train_dataset, function(x) sum(is.na(x)))

###emp_title - 29563 missing values
train_dataset$emp_title = factor(train_dataset$emp_title, 
                                 levels=c(levels(train_dataset$emp_title), 'Unemployed'))
train_dataset$emp_title[is.na(train_dataset$emp_title)] <- 'Unemployed'

###emp_length - 22615 missing values
train_dataset$emp_length[is.na(train_dataset$emp_length)] <- '< 1 year'

###dti - 148 missing values
mean(train_dataset$dti, na.rm = TRUE)
train_dataset$dti[is.na(train_dataset$dti)] <- mean(train_dataset$dti, na.rm = TRUE)

###inq_last_6mths - 1 missing values
mean(train_dataset$inq_last_6mths, na.rm = TRUE)
train_dataset$inq_last_6mths[is.na(train_dataset$inq_last_6mths)] <- mean(train_dataset$inq_last_6mths, na.rm = TRUE)

###revol_util - 213 missing values
mean(train_dataset$revol_util, na.rm = TRUE)
train_dataset$revol_util[is.na(train_dataset$revol_util)] <- mean(train_dataset$revol_util, na.rm = TRUE)

sapply(train_dataset, function(x) sum(is.na(x)))



###Test Dataset

str(test_dataset)
sapply(test_dataset, function(x) sum(is.na(x)))

### dropping the row where 'loan_amnt' has missing values since 180 rows are not correctly separated
### due to values in emp_title with double quotes which contain commas
test_dataset <- test_dataset[complete.cases(test_dataset$loan_amnt), ]

sapply(test_dataset, function(x) sum(is.na(x)))

###emp_title - 1224 missing values
test_dataset$emp_title = factor(test_dataset$emp_title, 
                                 levels=c(levels(test_dataset$emp_title), 'Unemployed'))
test_dataset$emp_title[is.na(test_dataset$emp_title)] <- 'Unemployed'

###emp_length - 1210 missing values
test_dataset$emp_length[is.na(test_dataset$emp_length)] <- '< 1 year'

###dti - 18 missing values
test_dataset$dti <- as.numeric(test_dataset$dti)
mean(test_dataset$dti, na.rm = TRUE)
test_dataset$dti[is.na(test_dataset$dti)] <- mean(test_dataset$dti, na.rm = TRUE)

###revol_util - 17 missing values
test_dataset$revol_util <- as.numeric(test_dataset$revol_util)
mean(test_dataset$revol_util, na.rm = TRUE)
test_dataset$revol_util[is.na(test_dataset$revol_util)] <- mean(test_dataset$revol_util, na.rm = TRUE)


sapply(test_dataset, function(x) sum(is.na(x)))


# ----------------------------------------------------------------------------

##Outliers
boxplot(train_dataset) 
str(train_dataset)

numerical_columns <- sapply(train_dataset, is.numeric)

#print the boxplot for the numerical features
for (col in names(train_dataset[, numerical_columns])) {
  boxplot(train_dataset[, col], 
          main = col, 
          col = "skyblue", 
          border = "black")
}

#print the histogram for the numerical features
for (col in names(train_dataset[, numerical_columns])) {
  hist(train_dataset[, col], 
       main = col, 
       col = "skyblue", 
       border = "black", 
       xlim = c(min(train_dataset[, col]), max(train_dataset[, col])),
       breaks = 30)
}

# for categorical features
categorical_columns <- sapply(train_dataset, is.factor)
# exclude emp_title, issue_d and earliest_cr_line due to the high number of levels
exclude_features <- c("emp_title", "issue_d", "earliest_cr_line")

# Identify the indices of the categorical columns to keep
categorical_columns <- !(names(train_dataset) %in% exclude_features) & categorical_columns

# Loop through each categorical column and plot bar charts
for (col in names(train_dataset[, categorical_columns])) {
  barplot(table(train_dataset[, col]), 
          main = col, 
          col = "skyblue",
          xlab = "Categories",
          ylab = "Count")
}


# Define threshold values
threshold_values <- list(
  int_rate = 25,
  annual_inc = 0.3 * 10^6,
  dti = 50,
  delinq_2yrs = 5,
  inq_last_6mths = 3,
  open_acc = 30,
  pub_rec = 3,
  revol_bal = 60000,
  revol_util = 110,
  total_acc = 70,
  out_prncp = 10000,
  total_pymnt = 50000,
  credit_line_duration = 20000,
  income_to_loan_ratio = 40
)

# Loop through each column and apply the corresponding threshold for the Train Dataset
for (column in names(train_dataset)) {
  if (column %in% names(threshold_values)) {
    train_dataset <- train_dataset[train_dataset[, column] <= threshold_values[[column]], ]
  }
}

# Loop through each column and apply the corresponding threshold for the Test Dataset
for (column in names(test_dataset)) {
  if (column %in% names(threshold_values)) {
    test_dataset <- test_dataset[test_dataset[, column] <= threshold_values[[column]], ]
  }
}

#print the boxplot for the numerical features after dropped the outliers
for (col in names(train_dataset[, numerical_columns])) {
  boxplot(train_dataset[, col], 
          main = col, 
          col = "skyblue", 
          border = "black")
}

#print the histogram for the numerical features after dropped the outliers
for (col in names(train_dataset[, numerical_columns])) {
  hist(train_dataset[, col], 
       main = col, 
       col = "skyblue", border = "black", 
       xlim = c(min(train_dataset[, col]), max(train_dataset[, col])),
       breaks = 30)
}

# ----------------------------------------------------------------------------
#Feature Engineering
# ----------------------------------------------------------------------------

# Select the columns with numerical data
numerical_columns <- sapply(train_dataset, is.numeric)

# Create a correlation matrix
correlation_matrix <- cor(train_dataset[, numerical_columns])
correlation_matrix

# Plot the correlation matrix heatmap
corrplot(correlation_matrix, 
         method = "color", 
         tl.col = "black", 
         col = COL1('YlGn'), 
         addCoef.col = "black", 
         tl.cex = 0.7, 
         number.cex = 0.4)

#drop high correlated features (>|0.7|)
train_dataset <- train_dataset[, -which(names(train_dataset) %in% c("funded_amnt", "funded_amnt_inv", "installment"))]
test_dataset <- test_dataset[, -which(names(test_dataset) %in% c("funded_amnt", "funded_amnt_inv", "installment"))]


### Feature Transformation
str(train_dataset)
str(test_dataset)

# removing the dot in total_pymnt. to match train dataset
colnames(test_dataset)[colnames(test_dataset) == 'total_pymnt.'] <- 'total_pymnt'
# Convert total_pymnt into numeric
test_dataset$total_pymnt <- as.numeric(test_dataset$total_pymnt)


# Convert emp_length into numeric
unique(train_dataset$emp_length)
mapping <- c(
  "< 1 year" = 0,
  "1 year" = 1,
  "2 years" = 2,
  "3 years" = 3,
  "4 years" = 4,
  "5 years" = 5,
  "6 years" = 6,
  "7 years" = 7,
  "8 years" = 8,
  "9 years" = 9,
  "10+ years" = 10
)

train_dataset <- train_dataset %>%
  mutate(emp_length = case_when(
    emp_length %in% names(mapping) ~ mapping[as.character(emp_length)],
    TRUE ~ NA_real_
  ))
train_dataset$emp_length <- as.numeric(train_dataset$emp_length)
unique(train_dataset$emp_length)

test_dataset <- test_dataset %>%
  mutate(emp_length = case_when(
    emp_length %in% names(mapping) ~ mapping[as.character(emp_length)],
    TRUE ~ NA_real_
  ))
test_dataset$emp_length <- as.numeric(test_dataset$emp_length)
unique(test_dataset$emp_length)

# Convert term into numeric
train_dataset$term <- as.numeric(gsub("[^0-9]+", "", train_dataset$term))
test_dataset$term <- as.numeric(gsub("[^0-9]+", "", test_dataset$term))


# Transform the levels for the categoric features (except emp_title and earliest_cr_line) into numeric
unique(train_dataset$grade)
train_dataset$grade <- as.numeric(factor(train_dataset$grade, 
                                         levels = levels(train_dataset$grade)))

unique(test_dataset$grade)
test_dataset$grade <- as.numeric(factor(test_dataset$grade, 
                                         levels = levels(test_dataset$grade)))



unique(train_dataset$addr_state)
train_dataset$addr_state <- as.numeric(factor(train_dataset$addr_state,
                                              levels = levels(train_dataset$addr_state)))

unique(test_dataset$addr_state)
test_dataset$addr_state <- as.numeric(factor(test_dataset$addr_state, 
                                              levels = levels(test_dataset$addr_state)))


unique(train_dataset$home_ownership)
train_dataset$home_ownership <- as.numeric(factor(train_dataset$home_ownership,
                                                  levels = levels(train_dataset$home_ownership)))
unique(test_dataset$home_ownership)
test_dataset$home_ownership <- as.numeric(factor(test_dataset$home_ownership, 
                                                  levels = levels(test_dataset$home_ownership)))


unique(train_dataset$verification_status)
train_dataset$verification_status <- as.numeric(factor(train_dataset$verification_status,
                                                       levels = levels(train_dataset$verification_status)))

unique(test_dataset$verification_status)
test_dataset$verification_status <- as.numeric(factor(test_dataset$verification_status, 
                                                       levels = levels(test_dataset$verification_status)))


unique(train_dataset$purpose)
train_dataset$purpose <- as.numeric(factor(train_dataset$purpose, 
                                           levels = levels(train_dataset$purpose)))
unique(test_dataset$purpose)
test_dataset$purpose <- as.numeric(factor(test_dataset$purpose, 
                                           levels = levels(test_dataset$purpose)))


# Create 'employment_stability' feature
train_dataset$employment_stability <- ifelse(train_dataset$emp_length >= 5, 1, 0)
test_dataset$employment_stability <- ifelse(test_dataset$emp_length >= 5, 1, 0)

# Create 'income_to_loan_ratio' feature
train_dataset$income_to_loan_ratio <- train_dataset$annual_inc/train_dataset$loan_amnt
test_dataset$annual_inc <- as.numeric(test_dataset$annual_inc)
test_dataset$income_to_loan_ratio <- test_dataset$annual_inc/test_dataset$loan_amnt


## To convert earliest_cr_line into numeric, we modify the feature computing the diff with the current month
# Convert the 'earliest_cr_line' column to a Date type with custom format
train_dataset$earliest_cr_line <- dmy(paste0("01-", train_dataset$earliest_cr_line))
unique(train_dataset$earliest_cr_line)
# Adjust two-digit years using if_else
train_dataset$earliest_cr_line <- if_else(year(train_dataset$earliest_cr_line) <= 2024,
                                          train_dataset$earliest_cr_line,
                                          train_dataset$earliest_cr_line - years(100))
unique(train_dataset$earliest_cr_line)

test_dataset$earliest_cr_line <- as.Date(test_dataset$earliest_cr_line, origin = "1899-12-30")
unique(train_dataset$earliest_cr_line)
unique(test_dataset$earliest_cr_line)

# Calculate the difference in months between each date and the reference date
current_date <- dmy("01-01-2024")
train_dataset$earliest_cr_line <- as.numeric(interval(train_dataset$earliest_cr_line, current_date) / months(1))
test_dataset$earliest_cr_line <- as.numeric(interval(test_dataset$earliest_cr_line, current_date) / months(1))
unique(train_dataset$earliest_cr_line)
unique(test_dataset$earliest_cr_line)

unique(train_dataset$issue_d)
unique(test_dataset$issue_d)

# Convert numerical labels to Date type 
test_dataset$issue_d <- as.Date(test_dataset$issue_d, origin = "1899-12-30")
unique(test_dataset$issue_d)
# since levels of issue_d between train and test are different, we cannot use label encoding
# thus, we modify the issue_d feature into the diff in months between issue_d and current month

# Convert the date_column to a Date type
train_dataset$issue_d <- dmy(paste0("01-", train_dataset$issue_d))
unique(train_dataset$issue_d)

current_date <- dmy("01-01-2024")

# Calculate the difference in months between each date and the current date
train_dataset$issue_d <- as.numeric(interval(train_dataset$issue_d, current_date) / months(1))
test_dataset$issue_d <- as.numeric(interval(test_dataset$issue_d, current_date) / months(1))
unique(train_dataset$issue_d)
unique(test_dataset$issue_d)


# Display the result
str(train_dataset)
str(test_dataset)


### Plot the correlation between the target feature 'Risk' and the other features

# Select the columns with numerical data
numerical_columns <- sapply(train_dataset, is.numeric)

# Calculate the correlation with the target variable
correlation_with_target <- sapply(train_dataset[, numerical_columns], 
                                  function(x) cor(x, train_dataset$risk))

# Order correlations by absolute values in descending order
sorted_correlations <- sort(abs(correlation_with_target), decreasing = FALSE)

# Exclude the 'risk' variable
sorted_correlations <- sorted_correlations[!(names(sorted_correlations) %in% "risk")]

# Set larger plot margins to fit the y labels
par(mar = c(5, 10, 2, 2))

# Create a barplot with the correlations to 'risk'
barplot(sorted_correlations, horiz = TRUE, col = "skyblue",
        xlab = "Absolute Correlation Coefficient",
        las = 1)

# Set plot margins to default
par(mar = c(5, 4, 4, 2))

# Extract column names with correlation less than 0.01 and drop them
columns_to_drop <- names(correlation_with_target[abs(correlation_with_target) < 0.001]) #only addr_state meet this condition
train_dataset <- train_dataset[, !names(train_dataset) %in% columns_to_drop]
test_dataset <- test_dataset[, !names(test_dataset) %in% columns_to_drop]

#drop emp_title due to high dispersion 
#drop loan_status since represent our target feature 'risk'
#drop issue_d since if we include it we get for some reason a perfect performance
train_df <- train_dataset[, -which(names(train_dataset) %in% c("emp_title", "loan_status", "issue_d"))]
test_df <- test_dataset[, -which(names(test_dataset) %in% c("emp_title", "issue_d"))]

# Display the result
str(train_df)
str(test_df)


# ----------------------------------------------------------------------------
# Logistic regression 
# ----------------------------------------------------------------------------

#Breaking Data into Training and Test Sample
set.seed(123)

# Data splitting
index <- createDataPartition(train_df$risk, p = 2/3, list = FALSE)
train <- train_df[index, ]
test <- train_df[-index, ]

# Fit the logistic regression model
model_log <- glm(risk ~ ., 
             data = train, 
             family = binomial(link = "logit"))

# Summary of the model
summary(model_log)

# Make predictions on the test set
y_pred <- predict(model_log, newdata = test, type = "response")

# Convert probabilities to binary predictions (assuming a threshold of 0.5)
y_pred_binary <- ifelse(y_pred > 0.5, 1, 0)
y_pred_binary <- as.factor(y_pred_binary)

## Evaluate the model
test$risk <- as.factor(test$risk)
# Accuracy
accuracy <- confusionMatrix(data = y_pred_binary, 
                            reference = test$risk)$overall["Accuracy"]
cat("Accuracy:", accuracy, "\n")

# Classification Report
conf_matrix <- confusionMatrix(data = y_pred_binary, reference = test$risk)
print(conf_matrix)

# Confusion Matrix
conf_matrix <- as.table(conf_matrix)
conf_matrix <- prop.table(conf_matrix, 1)

# Plot the correlation matrix 
conf_matrix

#Plot ROC Curve
roccurve <- roc(test$risk ~ y_pred)
auc_score <- auc(roccurve)
cat("Logistic Regression AUC Score:", auc_score, "\n")

plot(roccurve, col='blue')
# Add AUC value as text annotation
text(0, 0, paste("AUC =", round(auc_score, 2)), col="blue", cex=1.2)



# ----------------------------------------------------------------------------
# Random Forest 
# ----------------------------------------------------------------------------

#Breaking Data into Training and Test Sample
set.seed(123)

# Data splitting
index <- createDataPartition(train_df$risk, p = 2/3, list = FALSE)
train <- train_df[index, ]
test <- train_df[-index, ]

train$risk <- as.factor(train$risk)

# Fit the Random Forest model
model_rf <- randomForest(risk ~ .,
                         data = train, 
                         ntree = 100)

# Make predictions on the test set
y_pred_rf <- predict(model_rf, type = "prob", test)

# Convert probabilities to binary predictions (assuming a threshold of 0.5)
y_pred_binary_rf <- ifelse(y_pred_rf[,2] > 0.5, 1, 0)
y_pred_binary_rf <- as.factor(y_pred_binary_rf)

## Evaluate the Random Forest model
test$risk <- as.factor(test$risk)

# Accuracy
accuracy_rf <- confusionMatrix(data = y_pred_binary_rf, reference = test$risk)$overall["Accuracy"]
cat("Random Forest Accuracy:", accuracy_rf, "\n")

# Classification Report
conf_matrix_rf <- confusionMatrix(data = y_pred_binary_rf, reference = test$risk)
print(conf_matrix_rf)

# Confusion Matrix
conf_matrix_rf <- as.table(conf_matrix_rf)
conf_matrix_rf <- prop.table(conf_matrix_rf, 1)

# Plot the confusion matrix
conf_matrix_rf

# Plot ROC Curve for Random Forest
roccurve_rf <- roc(test$risk ~ y_pred_rf[,2])
# AUC Score for Random Forest
auc_score_rf <- auc(roccurve_rf)
cat("Random Forest AUC Score:", auc_score_rf, "\n")
plot(roccurve_rf, col = 'blue')
# Add AUC value as text annotation
text(0, 0, paste("AUC =", round(auc_score_rf, 2)), col="blue", cex=1.2)


# ----------------------------------------------------------------------------
# Deep Neural Network
# ----------------------------------------------------------------------------

install_tensorflow(envname = "r-tensorflow")

tf$constant("Hello TensorFlow!")

#Breaking Data into Training and Test Sample
set.seed(123)

# Data splitting
index <- createDataPartition(train_df$risk, p = 2/3, list = FALSE)
train <- train_df[index, ]
test <- train_df[-index, ]

y_train <- as.numeric(train$risk)
y_test <- as.numeric(test$risk)
x_train <- train %>% select(-risk)
x_test <- test %>% select(-risk)

# Normalize hte input features
x_train <- as.matrix(apply(x_train, 2, function(x) (x-min(x))/(max(x) - min(x))))
x_test <- as.matrix(apply(x_test, 2, function(x) (x-min(x))/(max(x) - min(x))))


# Fit the Neural Network model
model_nn <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(train) - 1) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

compile(model_nn, optimizer = "adam", 
        loss = "binary_crossentropy",
        metrics = c("accuracy"))

# Train the model
history <- fit(
  model_nn,
  x = x_train,
  y = y_train,
  shuffle = T,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

# Make predictions on the test set
y_pred_prob_nn <- predict(model_nn, x_test)

# Convert probabilities to binary predictions
y_pred_binary_nn <- ifelse(y_pred_prob_nn > 0.5, 1, 0)
y_pred_binary_nn <- as.factor(y_pred_binary_nn)

# Evaluate the neural network model
y_test <- as.factor(y_test)
# Accuracy
accuracy_nn <- confusionMatrix(data = y_pred_binary_nn, reference = y_test)$overall["Accuracy"]
cat("Neural Network Accuracy:", accuracy_nn, "\n")

# Classification Report
conf_matrix_nn <- confusionMatrix(data = y_pred_binary_nn, reference = y_test)
print(conf_matrix_nn)

# Confusion Matrix
conf_matrix_nn <- as.table(conf_matrix_nn)
conf_matrix_nn <- prop.table(conf_matrix_nn, 1)

# Plot the confusion matrix 
conf_matrix_nn

# Plot ROC Curve for Neural Network
roccurve_nn <- roc(y_test, as.numeric(y_pred_prob_nn))
# AUC Score for Neural Network
auc_score_nn <- auc(roccurve_nn)
cat("Neural Network AUC Score:", auc_score_nn, "\n")
plot(roccurve_nn, col = 'blue')
# Add AUC value as text annotation
text(0, 0, paste("AUC =", round(auc_score_nn, 2)), col="blue", cex=1.2)


# ----------------------------------------------------------------------------
# Deployment
# ----------------------------------------------------------------------------
# Based on the evaluation metrics, Random Forest is the model which perform better

# Make predictions on the unseen file using the trained model
unseen_df <- test_df[, !(names(test_df) %in% c("id"))]
y_pred_rf <- predict(model_rf, type = "prob", unseen_df)

# Convert probabilities to binary predictions (assuming a threshold of 0.5)
y_pred_binary_rf <- ifelse(y_pred_rf[,2] > 0.5, 1, 0)
y_pred_binary_rf <- as.factor(y_pred_binary_rf)

RandomForest <- data.frame(id=test_df$id, risk=y_pred_binary_rf)
RandomForest

write.csv(RandomForest,"Group5.csv",row.names = FALSE)

