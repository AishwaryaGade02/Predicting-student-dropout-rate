library(dplyr)
library(ggplot2)
library(gridExtra)
library(caret)

setwd("C:/Users/aishw/OneDrive/Documents/Courses/Predictive Modelling")
data <- read.csv("C:/Users/aishw/OneDrive/Documents/Courses/Predictive Modelling/predict+students+dropout+and+academic+success/data.csv", sep=";")

nrow(data)
ncol(data)
colnames(data)

##Categorical Variables : Marital.status, Application.mode, Application.order
#Course, Daytime.evening.attendance., Previous.qualification, Nacionality,
#Mother.s.qualification, Father.s.qualification, Mother.s.occupation,
#Father.s.occupation, Displaced, Educational.special.needs, Debtor,
#Tuition.fees.up.to.date, Gender, Scholarship.holder, International


#Categorical_var<- c("Marital.status", "Application.mode",
#                    "Course","Daytime.evening.attendance.", "Previous.qualification",
#                    "Nacionality","Mother.s.qualification", "Father.s.qualification",
#                    "Mother.s.occupation","Father.s.occupation","Displaced","Educational.special.needs",
#                    "Debtor","Tuition.fees.up.to.date","Gender","Scholarship.holder",
#                    "International")
Categorical_var<- c("Marital.status", "Application.mode","Application.order",
                    "Course","Daytime.evening.attendance.", "Previous.qualification",
                    "Nacionality","Mother.s.qualification", "Father.s.qualification",
                    "Mother.s.occupation","Father.s.occupation","Displaced","Educational.special.needs",
                    "Debtor","Tuition.fees.up.to.date","Gender","Scholarship.holder",
                    "International")






predictors <- data[, !colnames(data) %in% "Target"]

for(var in Categorical_var) {  
  if(is.numeric(data[[var]])) {
    data[[var]] <- as.factor(data[[var]])
  }
}

dummies <- dummyVars(as.formula(paste("~ ", paste(Categorical_var, collapse = " + "))), data = data)
dummy_data <- predict(dummies, newdata = data)

dummy_data <- as.data.frame(dummy_data)
dummy_data
columns_to_keep <- setdiff(names(data), Categorical_var)

data <- cbind(data[columns_to_keep], dummy_data)

ncol(data)
colnames(data)
nzv<-nearZeroVar(data)
length(nzv)
data <- data[,-nzv]
ncol(data)
colnames(data)


resposne <- data$Target
data<-data[, !colnames(data) %in% "Target"]

correlations<-cor(data)

library(corrplot)
win.graph(width=10, height=10,pointsize=8)
corrplot(correlations, order = "hclust")

correlations<-cor(data[, !colnames(data) %in% "Target"])
highCorr <- findCorrelation(correlations, cutoff = .85)
length(highCorr)
data<-data[,-highCorr]
ncol(data)
data

highCorr


#data["Target"] <- resposne
library(e1071)
skewness_values <- sapply(data,skewness)
print(skewness_values)

highly_skewed <- names(skewness_values[abs(skewness_values) > 0.5])
highly_skewed
length(highly_skewed)
colnames(data)
data

for (predictor in highly_skewed) {
  data[[predictor]] <- data[[predictor]] + 0.0001
}
#data$Curricular.units.1st.sem..grade.<-data$Curricular.units.1st.sem..grade.+0.0001
#data$Curricular.units.2nd.sem..grade.<-data$Curricular.units.2nd.sem..grade.+0.0001
data

#boxcox_trans <- c("Curricular.units.1st.sem..grade.","Curricular.units.2nd.sem..grade.")
xx1 <- preProcess(data[,highly_skewed], method = c("BoxCox"))

transformed <- predict(xx1, data[,highly_skewed])
transformed

data[, highly_skewed] <- transformed
ncol(data)




skewness_values <- sapply(data,skewness)
print(skewness_values)

data["Target"] <- resposne
data["Target"]


train.index <- createDataPartition(data[["Target"]], p = .8, list = FALSE)

predictors <- data[, !colnames(data) %in% "Target"]
class <- data$Target
X_train <- predictors[ train.index,]
X_test  <- predictors[-train.index,]
ncol(X_test)


y_train<-class[train.index]
y_test<-class[-train.index]
y_test
table(y_train)

win.graph(width=10, height=10,pointsize=10)
barplot(table(y_train),
        main = "Distribution of y_train",
        xlab = "Class",
        ylab = "Count",
        col = "pink",
        border = "black")
win.graph(width=10, height=10,pointsize=10)
barplot(table(y_test),
        main = "Distribution of y_test",
        xlab = "Class",
        ylab = "Count",
        col = "pink",
        border = "black")

ctrl <- trainControl(#method = "LGOCV",
                     method = "cv",
                     number = 10,
                     summaryFunction = multiClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)


lrFull <- train(X_train,
                y = y_train,
                method = "multinom",
                metric = "Kappa",
                preProc = c("center","scale","spatialSign"),
                trControl = ctrl)
lrFull

lrPredict <- predict(lrFull,X_test)
lrProb <- predict(lrFull, newdata = X_test,
                  type="prob")
y_test<-as.factor(y_test)

confusionMatrix(data = lrFull$pred$pred,
                reference = lrFull$pred$obs)

confusionMatrix(data = lrPredict,
                reference = y_test)

length(lrPredict)
lrPredict

install.packages("pheatmap")
library(pheatmap)

cm<-confusionMatrix(data = lrPredict,
                    reference = y_test)

cm_table <- cm$table

# Plot confusion matrix using pheatmap
win.graph(width=10, height=10,pointsize=10)
pheatmap(as.matrix(cm_table),
         display_numbers = TRUE,
         number_format = "%.0f",
         color = colorRampPalette(c("white", "blue"))(50),
         main = "Confusion Matrix",
         fontsize_number = 12)


##--------------LDA------------##
LDAFull <- train(X_train,
                 y = y_train,
                 method = "lda",
                 preProc = c("center","scale","spatialSign"),
                 metric = "Kappa",
                 trControl = ctrl)
LDAFull
confusionMatrix(data = LDAFull$pred$pred,
                reference = LDAFull$pred$obs)

LDAPred <- predict(LDAFull,
                   newdata = X_test)
LDAPred

LDAProbs <- predict(LDAFull, 
                    newdata = X_test,
                    type = "prob")
LDAProbs
LDAPred
y_test<-as.factor(y_test)
confusionMatrix(data = LDAPred,
                reference = y_test)## Error

##--------------PLS-LDA------------##
ncol(X_train)
plsFit2 <- train(X_train,
                 y = y_train,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:67),
                 preProc = c("center","scale","spatialSign"),
                 metric = "Kappa",
                 trControl = ctrl)


win.graph(width=10, height=10,pointsize=10)
plsFit2
plot(plsFit2)

PLSPred <- predict(plsFit2,
                   newdata = X_test)
PLSPred

PLSProbs <- predict(plsFit2, 
                    newdata = X_test,
                    type = "prob")
PLSProbs

confusionMatrix(data = plsFit2$pred$pred,
                reference = plsFit2$pred$obs)
confusionMatrix(data = PLSPred,
                reference = y_test)## Error

##----------Penalized Model----------##

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
set.seed(476)
glmnTuned <- train(x=X_train,
                   y = y_train,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale","spatialSign"),
                   metric = "Kappa",
                   trControl = ctrl)
glmnTuned

glmPred <- predict(glmnTuned,
                   newdata = X_test)
glmPred

glmProbs <- predict(glmnTuned, 
                    newdata = X_test,
                    type = "prob")
glmProbs

win.graph(width=10, height=10,pointsize=10)
plot(glmnTuned, plotType = "level")

confusionMatrix(data = glmnTuned$pred$pred,
                reference = glmnTuned$pred$obs)
confusionMatrix(data = glmPred,
                reference = y_test)
##----------Non Linear Models----------##
##-------QDA----------##
## If pca is not done, then qda can not execute
## If spatial sign and pca is done, then there is a error
table(y_train)
qdaFit <- train(x = X_train, 
                y = y_train,
                method = "qda",
                metric = "Kappa",
                preProc = c("center","scale","spatialSign"),
                trControl = ctrl)
qdaFit

win.graph(width=10, height=10,pointsize=10)
plot(qdaFit)



qdaPred <- predict(qdaFit,
                   newdata = X_test)
qdaPred

qdaProbs <- predict(qdaFit, 
                    newdata = X_test,
                    type = "prob")
qdaProbs


confusionMatrix(data = qdaFit$pred$pred,
                reference = qdaFit$pred$obs)
confusionMatrix(data = qdaPred,
                reference = y_test)

##-----------MDA-----------##


mdaFit <- train(x = X_train, 
                y = y_train,
                method = "mda",
                metric = "Kappa",
                preProc = c("center","scale","spatialSign"),
                tuneGrid = expand.grid(.subclasses = 1:20),
                trControl = ctrl)
mdaFit
win.graph(width=10, height=10,pointsize=10)
plot(mdaFit)

MDAPred <- predict(mdaFit,
                   newdata = X_test)
MDAPred

MDAProbs <- predict(mdaFit, 
                    newdata = X_test,
                    type = "prob")
MDAProbs

confusionMatrix(data = mdaFit$pred$pred,
                reference = mdaFit$pred$obs)
confusionMatrix(data = MDAPred,
                reference = y_test)
##-----------RDA-----------##
rdaGrid <- expand.grid(.gamma = seq(0, 1, by = 0.1),  
                       .lambda = seq(0.1, 1, by = 0.1)) 


rdaFit <- train(x = X_train,          
                y = y_train,          
                method = "rda",                    
                metric = "Kappa",  
                preProc = c("center","scale","spatialSign"),
                tuneGrid = rdaGrid,                
                trControl = ctrl)
rdaFit
win.graph(width=10, height=10,pointsize=10)
plot(rdaFit)

RDAPred <- predict(rdaFit,
                   newdata = X_test)
RDAPred

RDAProbs <- predict(rdaFit, 
                    newdata = X_test,
                    type = "prob")
RDAProbs

confusionMatrix(data = rdaFit$pred$pred,
                reference = rdaFit$pred$obs)
confusionMatrix(data = RDAPred,
                reference = y_test)

##-----------NN-----------##
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (58 + 1) + (maxSize+1)*2)
nnetFit <- train(x = X_train, 
                 y = y_train,
                 method = "nnet",
                 metric = "Kappa",
                 preProc = c("center", "scale", "spatialSign"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit
win.graph(width=10, height=10,pointsize=10)
plot(nnetFit)

NNPred <- predict(nnetFit,
                  newdata = X_test)
NNPred

NNProbs <- predict(nnetFit, 
                   newdata = X_test,
                   type = "prob")
NNProbs

confusionMatrix(data = nnetFit$pred$pred,
                reference = nnetFit$pred$obs)
confusionMatrix(data = NNPred,
                reference = y_test)


##-----------FDA-----------##

marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:70)
#marsGrid <- expand.grid(.degree = 1:2, .nprune = seq(2, min(10, nrow(X_train) - 1)))

fdaTuned <- train(x = X_train, 
                  y = y_train,
                  method = "fda",
                  preProc = c("center", "scale","spatialSign"),
                  metric = "Kappa",
                  # Explicitly declare the candidate models to test
                  tuneGrid = marsGrid,
                  trControl = ctrl)

fdaTuned

win.graph(width=10, height=10,pointsize=10)
plot(fdaTuned)

fdaPred <- predict(fdaTuned,
                   newdata = X_test)
fdaPred

fdaProbs <- predict(fdaTuned, 
                    newdata = X_test,
                    type = "prob")
fdaProbs

confusionMatrix(data = fdaTuned$pred$pred,
                reference = fdaTuned$pred$obs)
y_test<-as.factor(y_test)
confusionMatrix(data = fdaPred,
                reference = y_test)

##------------SVM-----------##
library(kernlab)
library(caret)
sigmaRangeReduced <- sigest(as.matrix(X_train))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 10)))
set.seed(476)
svmRModel <- train(x = X_train, 
                   y = y_train,
                   method = "svmRadial",
                   metric = "Kappa",
                   preProc = c("center", "scale","spatialSign"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = ctrl)
svmRModel
win.graph(width=10, height=10,pointsize=10)
plot(svmRModel)
ggplot(svmRModel)+coord_trans(x='log2')
SVMPred <- predict(svmRModel,
                   newdata = X_test)
SVMPred

SVMProbs <- predict(svmRModel, 
                    newdata = X_test,
                    type = "prob")
SVMProbs

confusionMatrix(data = svmRModel$pred$pred,
                reference = svmRModel$pred$obs)
confusionMatrix(data = SVMPred,
                reference = y_test)
##-----------KNN-----------##
knnFit <- train(x = X_train, 
                y = y_train,
                method = "knn",
                metric = "Kappa",
                preProc = c("center", "scale","spatialSign"),
                ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                tuneGrid = data.frame(.k = 1:67),
                trControl = ctrl)

knnFit
win.graph(width=10, height=10,pointsize=10)
plot(knnFit)

KNNPred <- predict(knnFit,
                   newdata = X_test)
KNNPred

KNNProbs <- predict(knnFit, 
                    newdata = X_test,
                    type = "prob")
KNNProbs

confusionMatrix(data = knnFit$pred$pred,
                reference = knnFit$pred$obs)
confusionMatrix(data = KNNPred,
                reference = y_test)



##-----------NB-----------##


library(klaR)
set.seed(476)
nbFit <- train( x = X_train, 
                y = y_train,
                method = "nb",
                metric = "Kappa",
                preProc = c("center", "scale","spatialSign"),
                #tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
                trControl = ctrl)

nbFit
win.graph(width=10, height=10,pointsize=10)
plot(nbFit)


NBPred <- predict(nbFit,
                  newdata = X_test)
NBPred

NBProbs <- predict(nbFit, 
                   newdata = X_test,
                   type = "prob")
NBProbs

confusionMatrix(data = nbFit$pred$pred,
                reference = nbFit$pred$obs)
confusionMatrix(data = NBPred,
                reference = y_test)


##-------------Variance Imp Plot-------------##
importance <- varImp(fdaTuned, scale = TRUE)

imp_df <- data.frame(
  Feature = rownames(importance$importance),
  Importance = importance$importance[,1]
)


imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE),]
imp_df
imp_df <- imp_df[1:10,]
win.graph(width=10, height=10,pointsize=10)
ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_point() +
  geom_segment(aes(x = Feature, xend = Feature, y = 0, yend = Importance)) +
  coord_flip() +
  theme_minimal() +
  labs(x = "", y = "Importance", title = "Variable Importance") +
  theme(axis.text.y = element_text(size = 8))

