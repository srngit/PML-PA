library(caret)
train <- read.csv("pml-training.csv")
test  <- read.csv("pml-testing.csv")

train <- train[ , -colSums(is.na(train)) == 0]

## Near Zero Variance

nzv <- nearZeroVar(train,saveMetrics=TRUE)
nzvcols <- rownames(nzv)[nzv$nzv==TRUE]
train <- train[ , -which(names(train) %in% nzvcols)]
train <- train[ , -c(1:6)]
    
## creating Data Partition
set.seed(1234)
InTrain <- createDataPartition(train$classe,p=0.75,list=FALSE)
subtrain   <- train[InTrain, ]
subtest    <- train[-InTrain, ]

## PCA
prComp <- prcomp(train[,-53])
plot(cumsum(prComp$sdev^2/sum(prComp$sdev^2)), xlab = "Variables", ylab="Cumulative Var")
pr2 <- prcomp(train[,-53], tol=0.1)
od=prComp$x %*% t(prComp$rotation)
od2=pr2$x %*% t(pr2$rotation)


## Random Forest 
library(randomForest)
set.seed(1234)
rfFit <- randomForest(factor(classe) ~., data=subtrain, importance =TRUE)
rfPred <- predict(rfFit, newdata=subtest)
confusionMatrix(rfPred,subtest$classe)

## rf Train function
rfFit2  <- train(factor(classe) ~., data=subtrain, method="rf", prox=TRUE)
rfPred2  <- predict(rfFit2, newdata=subtest)
confusionMatrix(rfPred,subtest$classe)

## Boosting
gbmFit <- train(factor(classe) ~., data=subtrain[ , -c(1:6)], method="gbm", verbose=FALSE)


## Aplying the model on Test Data
fpred <- predict(rfFit, newdata=test)
fpred