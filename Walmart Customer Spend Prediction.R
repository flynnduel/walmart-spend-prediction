library(tidyverse)
library(caret)

# Load dataset
walmart <- read_csv("walmart.csv")

# View structure and missing values
summary(walmart)
colSums(is.na(walmart))

# Clean "Stay_In_Current_City_Years"
walmart <- walmart %>%
  mutate(Stay_In_Current_City_Years = str_replace(Stay_In_Current_City_Years, "\\+", ""),
         Stay_In_Current_City_Years = as.integer(Stay_In_Current_City_Years))

# Convert categorical variables to factors
walmart <- walmart %>%
  mutate(
    Gender = as.factor(Gender),
    Age = factor(Age, ordered = TRUE, 
                 levels = c("0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+")),
    Occupation = as.factor(Occupation),
    City_Category = as.factor(City_Category),
    Marital_Status = as.factor(Marital_Status)
  )

# Create dataset for Random Forest (factors kept)
walmart_rf <- walmart %>%
  select(-User_ID, -Product_ID)

# Create dataset for LASSO (dummy encoding)
walmart_lasso_base <- walmart_rf
dummies <- dummyVars(Purchase ~ ., data = walmart_lasso_base)
walmart_lasso <- data.frame(predict(dummies, newdata = walmart_lasso_base))

# Add target variable back
walmart_lasso$Purchase <- walmart_lasso_base$Purchase

# Optional: view distribution of Purchase
walmart %>% 
  ggplot(aes(Purchase)) +
  geom_histogram(bins = 50, fill = "blue", color = "white") +
  labs(title = "Distribution of Purchase Amounts")


set.seed(123)

# For Random Forest
trainIndex_rf <- createDataPartition(walmart_rf$Purchase, p = 0.7, list = FALSE)
train_rf <- walmart_rf[trainIndex_rf, ]
test_rf  <- walmart_rf[-trainIndex_rf, ]

# For LASSO
trainIndex_lasso <- createDataPartition(walmart_lasso$Purchase, p = 0.7, list = FALSE)
train_lasso <- walmart_lasso[trainIndex_lasso, ]
test_lasso  <- walmart_lasso[-trainIndex_lasso, ]

# Random Forest with 10-fold cross-validation
tunegrid_rf <- expand.grid(mtry = c(4, 6, 8, 10))
control_rf <- trainControl(method = "cv", number = 10)

set.seed(123)
rf_model <- train(Purchase ~ ., 
                  data = train_rf,
                  method = "rf",
                  tuneGrid = tunegrid_rf,
                  trControl = control_rf,
                  importance = TRUE)

# Best mtry value
rf_model$bestTune

# Predict on test set
pred_rf <- predict(rf_model, test_rf)

# Evaluate performance
postResample(pred = pred_rf, obs = test_rf$Purchase)

# Variable importance
varImpPlot(rf_model$finalModel, main = "Variable Importance - Random Forest")

# LASSO with 10-fold cross-validation
control_lasso <- trainControl(method = "cv", number = 10)
grid_lasso <- expand.grid(alpha = 1, lambda = 10^seq(3, -3, length = 100))

set.seed(123)
lasso_model <- train(Purchase ~ ., 
                     data = train_lasso,
                     method = "glmnet",
                     tuneGrid = grid_lasso,
                     trControl = control_lasso)

# Best lambda
lasso_model$bestTune

# Predict on test set
pred_lasso <- predict(lasso_model, test_lasso)

# Evaluate performance
postResample(pred = pred_lasso, obs = test_lasso$Purchase)

# View coefficient estimates
coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
