#------------------------------------------------------------------------------#
#   Name: Rashawn Edwards
#   Homework 2: k-NN Classification of Titanic Survivors
#------------------------------------------------------------------------------#

#Set Directory
setwd("C:/Users/medma/Documents/MBA/Data Mining/rcode")

#---------- DATA UNDERSTANDING -------------------------------------------
#Retrieve Data
titanic <- read.csv("../data/Titanic_knn.csv")

#Convert column from integer into factor
titanic$survived <- factor(titanic$survived)

#Bar Plots
#Class vs. Passengers
barplot(table(titanic$pclass),
        main = "Count by Passenger Class",
        xlab = "Count",
        ylab = "Number of Passengers")

#Survived vs. Passengers
barplot(table(titanic$survived),
        main = "Survivor Count",
        xlab = "Survived? (1=yes)",
        ylab = "Number of Passengers")

#Survived vs. Passengers
barplot(table(titanic$sex),
        main = "Passengers by Gender",
        xlab = "Female=0 Male=1",
        ylab = "Number of Passengers")

#---------- DATA PREPARATION ---------------------------------------------
#---------- DATA PREPARATION ---------------------------------------------
# Normalize the measurement data
# Create a normalization function to transform the values.
normalize <- function(x) {return ((x-min(x))/(max(x)-min(x)))}

#Normalization of bridge data
titanic_n <- as.data.frame(lapply(titanic[2:5], normalize))

#Training and Testing Datasets from Normalized Data
train <- titanic_n[1:1209,]
test <- titanic_n[1210:1309,]

#Training and Testing Labels
cl <- titanic[1:1209,1]
test_labels <- titanic[1210:1309,1]

#---------- MODEL BUILDING -----------------------------------------------
#Load "class" package
library(class)

#k-NN Test
pred <- knn(train, test, cl, k = 35, prob = TRUE)
#K value is the closest odd number to the square root of 1209.

#---------- TESTING AND EVALUATION ---------------------------------------
#Load "gmodels" package
library(gmodels)

#CrossTable
CrossTable(x=test_labels,
           y=pred,
           prop.chisq = FALSE)

#---------- PREDICTION PROBABILITIES ---------------------------------------
# Get the prediction probabilities
pred_prob <- attr(pred, "prob")

# Combine prediction, true values, and probabilities in one dataframe
results <- data.frame(test_labels, pred, pred_prob)

#Load caret and e1071 packages
library(caret)

#Confusion Matrix
confusionMatrix(results$pred, results$test_labels, positive = "1")

#Conclusion
#Though the model has an accuracy of 72%, its Kappa stat. of 0.3569 would
#suggest that the model's predictive capacity is only fair. It's precision
#value of 0.5806 also suggests that the model's ability to predict positive
#values correctly is moderate. The model is moderately good overall.