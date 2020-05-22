# TITANIC SURVIVAL PREDICTION 

## Data Tunning

# Getting and Setting in the Working Directory.
getwd()
setwd("C:/Users/RAJ/Desktop/Sem-2/R")

# Loading the data which is in 'xls' format using read_excel.
library("readxl")
titanic<-read_excel("Titanic.xls")

# Top Entries of data
head(titanic)

# Checking the summary of data
summary(titanic)

# Checking the structure of data
str(titanic)

# Checking the dimensions of data
dim(titanic)

# Removing all the varibles that are not required using Null
titanic$PassengerId<-NULL
titanic$Cabin<-NULL
titanic$Name<-NULL
titanic$Ticket<-NULL

# Converting necessary variables to factor.
titanic$Survived<-factor(titanic$Survived)
titanic$Pclass<-factor(titanic$Pclass)
titanic$SibSp<-factor(titanic$SibSp)
titanic$Parch<-factor(titanic$Parch)
titanic$Sex<-factor(titanic$Sex)

## Na's Imputation

# Checking the Na's in data.
colSums(is.na(titanic))

# Replacing Na's in Age with median
median(titanic$Age,na.rm = T) # na.rm T removes all NA from data
summary(titanic$Age) # we will use median to replace missing values
titanic$Age[is.na(titanic$Age)]<-28

# Splitting the age into differnt ranges.
titanic$Age<-cut(titanic$Age,breaks = c(0,20,28,40,Inf),labels = c("c1","c2","c3","c4"))

# Replacing Na's in Embarrked with mode.
summary(titanic$Embarked)
titanic$Embarked[is.na(titanic$Embarked)]<-"S"
titanic$Embarked<-factor(titanic$Embarked)
summary(titanic$Embarked)

# Again checking for null values.
colSums(is.na(titanic))

## Visualization

# Plotting the number of Survived Passengers
library(tidyr)
library(dplyr)
library(ggplot2)

# Plotting number of Survived passengers with respect to Pclass
titanic%>%ggplot(aes(x=Pclass,fill=factor(Survived)))+geom_bar(stat="count",position="fill")

# Plotting number of Survived passengers with respect to Sex
titanic%>%ggplot(aes(x=Sex,fill=factor(Survived) ))+geom_bar(stat = "count",position="fill")

# Pairplots
pairs(titanic)

# Making copy of data to apply different algorithms.
titanic_cp<-  titanic[sample(nrow(titanic)),] # for logistic regression

## Splitting data into training and testing data
index<-sample(nrow(titanic_cp),0.70*nrow(titanic_cp))
train<-titanic_cp[index,] 
test<-titanic_cp[-index,]
dim(train)
dim(test)

## BINARY LOGISTIC REGRESSION

# applying binary logistics to training data
titanic_model<-glm(Survived ~.,family=binomial(link='logit'),data=train)
titanic_model

### find cutoff using ROC Curve 
library(ROCR)
test_pred1 <- predict(titanic_model,newdata=test,type='response')
ROCRpred <- prediction(test_pred1, test$Survived)
ROCRperf <- performance(ROCRpred, measure = "tpr", x.measure = "fpr")
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7), print.cutoffs.at = seq(0,1,0.1))

# Using cutoff-0.35
test_pred1 <- ifelse(test_pred1 > 0.35,1,0)
str(test_pred1)
summary(test_pred1)

# Converting predictions to factor.
test_pred1 <- factor(test_pred1)
str(test_pred1)

# Checking Performance parameters using Confusion matrix
library(caret)
R1 <- confusionMatrix(data=test_pred1, reference=test$Survived)
R1

## Calculating the area under the curve(AUC)
auc <- performance(ROCRpred, measure = "auc")
auc <- auc@y.values[[1]]
auc

## NAIVE BAYES

library(e1071)
Nb_model <- naiveBayes(Survived~., data = train)
Nb_model

# Predicting test data.
test_pred2 <- predict(Nb_model,newdata = test)

# Checking Performance parameters using Confusion matrix
R2 <- confusionMatrix(test_pred2, test$Survived )
R2

## RANDOM FOREST

# Applying Random Forest on training data
library(randomForest)
rf_model <- randomForest(Survived~.,data=train, ntree=100, importance=TRUE)
rf_model

# Predicting test data
test_pred3 <- predict(rf_model,newdata = test)

# Checking Performance parameters using Confusion matrix
R3 <- confusionMatrix(test_pred3, test$Survived )
R3

## SUPPORT VECTOR MACHINE

# Applying Support Vector Machine
svm_model <- svm(Survived ~., data = train)
svm_model

# Predicting test data
test_pred4 <- predict(svm_model,newdata = test)

# Checking Performance parameters using Confusion matrix
R4 <- confusionMatrix(test_pred4, test$Survived )
R4

# The best Accuracy is obtained from Random Forest.
