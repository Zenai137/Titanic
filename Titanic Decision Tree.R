#Set Directory
getwd()
setwd("C:/Users/medma/Documents/MBA/Data Mining/rcode")

#---------- DATA UNDERSTANDING -------------------------------------------
#Retrieve Data
titanic <- read.csv("../data/Titanic_tree.csv", stringsAsFactors = TRUE)

#data structure
str(titanic)

#---------- DATA PREPARATION ---------------------------------------------
#create a random sample for training and test data
#use set.seed to use the same random number sequence for reproducibility
set.seed(101)
titanic_sample <- sample(1309,1009)

str(titanic_sample)

# split the data frames
train <- titanic[titanic_sample,]
test <- titanic[-titanic_sample,]

#---------- MODELING -----------------------------------------------------
# Install the C50 package, if necessary
# Load the C50 library
library(C50)

# build the simplest decision tree
# Using the negative index number excludes that column.
# That way we don't need intermediate data structures.
titanic_model <- C5.0(train[,c("age", "pclass", "sex", "sibsp",
                              "parch")], train$survived)

#display simple facts about the tree
titanic_model

#display detailed information about the tree
summary(titanic_model)

#Visualize the tree - not much use yet
plot(titanic_model)

#---------- EVALUATION & TESTING -----------------------------------------
#create a factor vector of predictions on test data
pred <- predict(titanic_model, test)
#model is applied to the test data

#cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(test$survived, pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual', 'predicted'))
#prop.c=columns, prop.r=rows, dnn=dimensions

#Confusion Matrix
library(caret)
library(e1071)
confusionMatrix(pred, test$survived, positive = "Y")

#Accuracy
#The model does a moderate job of predicting the fates of the other 300
#passengers, based on its kappa stat. of 58%, its sensitivity of 74%, and its
#specificity of 85%. 

#friends and family
titanic_ff <- read.csv("../data/titanic_ff.csv", stringsAsFactors = TRUE)

survived <- predict(titanic_model, titanic_ff)

fam_results <- data.frame(titanic_ff,survived)

#results commentary
#The most salient result to me is that all of the women survived, and all
#of the men died. It makes sense that the women would survive, since women
#and children were allowed on safety boats first.