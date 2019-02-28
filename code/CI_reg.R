library(RSNNS)
library(mlp)
library(data.table)
library(dplyr)
library(ggplot2)
library(stringr)
library(DT)
library(tidyr)
library(corrplot)
library(leaflet)
#install.packages("lubridate")
library(lubridate)
library(Metrics)
library(ROSE)
library(pROC)
library(caret)
library(caTools)
library(zoo)
library(neuralnet)
library(nnet)
library(NeuralNetTools)
library(MASS)
#install.packages("DataExplorer")
library(DataExplorer)
library(ISLR)
library(caTools) # sample.split
library(boot) # cv.glm
library(faraway) # compact lm summary "sumary" function
library(caret) # useful tools for machine learning
library(corrplot)
setwd('/Users/saurabhsemwal/Downloads/OnlineNewsPopularity')
dt=read.csv('OnlineNewsPopularity.csv')

#check for missing
sapply(dt, function(x) sum(is.na(x)))

#scaling data
dt<- as.data.frame(scale(dt))
min.shares <- min(dt$shares)
max.shares <- max(dt$shares)
# response var must be scaled to [0 < resp < 1]
dt$shares <- scale(dt$shares
                   , center = min.shares
                   , scale = max.shares - min.shares)


set_split <- sample(x = nrow(dt) , size = floor(0.8*nrow(dt)))
train_set_base <- dt[set_split,]
test_set_base <- dt[-set_split,]

#running on base model
n1 <- names(dt)
f1 <- as.formula(paste("shares ~", paste(n1[!n1 %in% c("shares")], collapse = " + ")))
nn <- neuralnet(f1,
                data = train_set_base,
                hidden = c(10),
                linear.output = TRUE,
                lifesign = "minimal")
plot(nn)
nn.results<-compute(nn,test_set_base[-60])
prediction_base = nn.results$net.result
RMSE(prediction_base,test_set$shares)


#####check for near zero variance
nzv_cols <- nearZeroVar(dt)
#removing the column with near zero variance
if(length(nzv_cols) > 0) dt <- dt[, -nzv_cols]
######checking correlations
M <- cor(dt)
corrplot(cor(dt), method = "number")

#######checking for linearity in the data
findLinearCombos(dt)
#removing 37th column, which is the weekend indicator. it can be derived from sat and sunday indicators
dt<-dt[, -c(36,37)]


set_split <- sample(x = nrow(dt) , size = floor(0.8*nrow(dt)))
train_set <- dt[set_split,]
test_set <- dt[-set_split,]


# neuralnet doesn't accept resp~. (dot) notation
# so a utility function to create a verbose formula is used
n <- names(dt)
f <- as.formula(paste("shares ~", paste(n[!n %in% c("shares")], collapse = " + ")))

#neuralnet package using resilient backpropagation with weight backtracking as its standard algorithm.
nn <- neuralnet(f,
                data = train_set,
                hidden = c(10,10),
                linear.output = TRUE,
                lifesign = "minimal", rep = 2, err.fct = "sse", algorithm = "rprop+")
plot(nn)

nn.results.test<-compute(nn,test_set[-58])
nn.results.train<-compute(nn,train_set[-58])
prediction_nn_test = nn.results.test$net.result
prediction_nn_train = nn.results.train$net.result
#calculating rmse for both test and train to check for overfit
RMSE(prediction_nn_test,test_set$shares)
RMSE(prediction_nn_train,train_set$shares)

#RBF architecture
model <- rbf(train_set, train_set$shares, size=40, maxit=1000,
             initFuncParams=c(0, 1, 0, 0.01, 0.01),
             learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8), linOut=TRUE)


prediction_rbf_test <- predict(model, test_set)
prediction_rbf_train <- predict(model, train_set)

RMSE(prediction_rbf_test,test_set$shares)
RMSE(prediction_rbf_train,train_set$shares)
###################
#applying pca
pca <- prcomp(dt[-58])
pca_out= predict(pca, dt[-58])

#visualizing
summary(pca)

#scree plot 
fviz_eig(pca)

#biploar plt
fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
#pca loading
loadings <- pca$rotation
print(loadings)
sdev<-pca$sdev
pr_var<-sdev^2
pr_var[1:10]
prop_var<-pr_var/sum(pr_var)
prop_var[1:20]
sum(prop_var[1:30])

pca = preProcess(x = dt[-58], method = 'pca', pcaComp = 30)
train_pca = predict(pca, train_set)
train_pca = train_pca[c(2:31, 1)]
test_pca = predict(pca, test_set)
test_pca = test_pca[c(2:31 , 1)]


model <- rbf(train_pca, train_pca$shares, size=40, maxit=1000,
             initFuncParams=c(0, 1, 0, 0.01, 0.01),
             learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8), linOut=TRUE)


prediction2 <- predict(model, test_pca)
RMSE(prediction2,test_pca$shares)


#nn
n <- names(train_pca)
f <- as.formula(paste("shares ~", paste(n[!n %in% c("shares")], collapse = " + ")))

#neuralnet package uses resilient backpropagation with weight backtracking as its standard algorithm.
nn <- neuralnet(f,
                data = train_pca,
                hidden = c(10,10),
                linear.output = TRUE,
                lifesign = "minimal", rep = 2, err.fct = "sse", algorithm = "rprop+")

nn.results<-compute(nn,test_pca[-31])
results<-data.frame(actual = test_pca$shares, prediction = nn.results$net.result)
prediction3 = nn.results$net.result
RMSE(prediction3,test_pca$shares)
###############################
#ENSEMBLE
#training rbf model on the error of the first model
#creating training and test data
train_rbf<-train_set
test_rbf<-test_set
train_rbf$shares<-train_set$shares - prediction_nn_train
test_rbf$shares<-test_set$shares - prediction_nn_test

model_rbf <- rbf(train_rbf, train_rbf$shares, size=40, maxit=1000,
             initFuncParams=c(0, 1, 0, 0.01, 0.01),
             learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8), linOut=TRUE)

prediction_rbf_test <- predict(model, test_rbf)
prediction_rbf_train <- predict(model, train_rbf)
#calculating rmse for both test and train to check for overfit
RMSE(prediction_rbf_test,test_rbf$shares)
RMSE(prediction_rbf_train,train_rbf$shares)
final.prediction <- prediction_nn_test + prediction_rbf_test 

error<-RMSE(final.prediction,test_set$shares)