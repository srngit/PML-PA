library(caret)
train <- read.csv("pml-training.csv")
test  <- read.csv("pml-testing.csv")

train <- train[ , -colSums(is.na(train)) == 0]

##common <- intersect(names(train),names(test))
##test <- test[ , -colSums(is.na(train)) == 0]

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
prComp <- prcomp(subtrain[,-53])
plot(cumsum(prComp$sdev^2/sum(prComp$sdev^2)))
prComp$sdev
plot(prComp)
barplot(prComp$sdev/prComp$sdev[1])
pr2 <- prcomp(subtrain[,-53], tol=0.1)
plot(pr2)
od=prComp$x %*% t(prComp$rotation)
od2=pr2$x %*% t(pr2$rotation)
plot(od)
plot(od2)
loadings(pr2)

plot(pr2$rotation[ ,2])
pr2


preProc <- preProcess(subtrain[ ,-53], method="pca", pcaComp=20)
rownames(preProc$rotation)

## Random Forest 
library(randomForest)
set.seed(1234)
rfFit <- randomForest(factor(classe) ~., data=subtrain, importance =TRUE)
rfPred <- predict(rfFit, newdata=subtest)
confusionMatrix(rfPred,subtest$classe)

## Boosting
rfFit  <- train(factor(classe) ~., data=subtrain, method="rf", prox=TRUE)
rfPred  <- predict(rfFit, newdata=subtest)
confusionMatrix(rfPred,subtest$classe)

gbmFit <- train(factor(classe) ~., data=subtrain[ , -c(1:6)], method="gbm", verbose=FALSE)


## Aplying the model on Test Data
fpred <- predict(rfFit, newdata=test)
plot(fpred)