library(mlr)
require(class)
library(klaR)

set.seed(50)
n.test <- 100
lab1 <- rep(0,n.test)
lab2 <- rep(1,n.test)
x11 <- rnorm(n.test,-0.5,1)
x12 <- rnorm(n.test,-0.5,1)
x1 <- cbind(x11,x12)
x21 <- rnorm(n.test,0.5,1)
x22 <- rnorm(n.test,0.5,1)
x2 <- cbind(x21,x22)
data.test <- data.frame(obs=matrix(c(x1,x2),byrow=TRUE,nrow=2*n.test),category=as.factor(c(lab1,lab2)))


set.seed(51)
n.train <- 250
lab1 <- rep(0,n.train)
lab2 <- rep(1,n.train)
y11 <- rnorm(n.train,-0.5,1)
y12 <- rnorm(n.train,-0.5,1)
y1 <- cbind(y11,y12)
y21 <- rnorm(n.train,0.5,1)
y22 <- rnorm(n.train,0.5,1)
y2 <- cbind(y21,y22)
data.train <- data.frame(obs=matrix(c(y1,y2),byrow=TRUE,nrow=2*n.train),category=as.factor(c(lab1,lab2)))

n.true <- 1000000
lab1 <- rep(0,n.true)
lab2 <- rep(1,n.true)
y11 <- rnorm(n.true,-0.5,1)
y12 <- rnorm(n.true,-0.5,1)
y1 <- cbind(y11,y12)
y21 <- rnorm(n.true,0.5,1)
y22 <- rnorm(n.true,0.5,1)
y2 <- cbind(y21,y22)
data.big <- data.frame(obs=matrix(c(y1,y2),byrow=TRUE,nrow=2*n.true),category=as.factor(c(lab1,lab2)))

task.train = makeClassifTask(id = "simulation", data = data.train, target = "category")
task.test = makeClassifTask(id = "simulation2", data = data.test, target = "category")
task.true = makeClassifTask(id = "trueerror", data=data.big, target="category")

mu1 <- c(-0.5,-0.5)
mu2 <- c(0.5,0.5)
sigma <- matrix(c(1,0,0,1),nrow=2,byrow=TRUE)

lrn = makeLearner("classif.knn",k=15)
model = train(learner=lrn, task=task.train)
data.pred = predict(object=model,task.true)
confmat = calculateConfusionMatrix(data.pred)
Err.true <- confmat$result[3,3]/(2*n.true)
data.pred.test <- predict(object=model,task.test)
confmat.test <- calculateConfusionMatrix(data.pred.test)
Err.test <- confmat.test$result[3,3]/(2*n.test)
data.pred.train <- predict(object=model,task.train)
confmat.train <- calculateConfusionMatrix(data.pred.train)
Err.train <- confmat.train$result[3,3]/(2*n.train)
loo <- makeResampleDesc("LOO")
res <- resample(lrn, task=task.train, resampling=loo)

n.test <- 20
lab1 <- rep(0,n.test)
lab2 <- rep(1,n.test)
z <- 100
K <- c(5,10,15,20,40)
h <- matrix(0,nrow=z,ncol=length(K))
err.train <- matrix(0,nrow=z,ncol=1)
err.test <- matrix(0,nrow=z,ncol=1)
err.true <- matrix(0,nrow=z,ncol=1)

for(j in 1:length(K)){
  for(i in 1:z){
    set.seed(i+3)
    x11 <- mvrnorm(n.test,mu1,sigma)
    #x12 <- mvrnorm(n.test,mu1,sigma)
    #x1 <- cbind(x11,x12)
    #x21 <- mvrnorm(n.test,mu2,sigma)
    x22 <- mvrnorm(n.test,mu2,sigma)
    #x2 <- cbind(x21,x22)
    m <- data.frame(obs=matrix(c(x11,x22),nrow=2*n.test,byrow=TRUE),category=as.factor(c(lab1,lab2)))
    task.train <- makeClassifTask(data=m, target="category")
    lrn=makeLearner("classif.knn",k=15)
    hi = makeResampleDesc("CV",iters=K[j])
    res = resample(lrn, task=task.train, resampling=hi)
    h[i,j]<-res$aggr
  }
}

for(i in 1:z){
  set.seed(i+3)
  lab1 <- rep(0,n.test)
  lab2 <- rep(1,n.test)
  x11 <- mvrnorm(n.test,mu1,sigma)
  #x12 <- mvrnorm(n.test,mu1,sigma)
  #x1 <- cbind(x11,x12)
  x21 <- mvrnorm(n.test,mu2,sigma)
  #x22 <- mvrnorm(n.test,mu2,sigma)
  #x2 <- cbind(x21,x22)
  m <- data.frame(obs=matrix(c(x11,x21),nrow=2*n.test,byrow=TRUE),category=as.factor(c(lab1,lab2)))
  task.train <- makeClassifTask(data=m, target="category")
  task.test <- makeClassifTask(data=data.test,target="category")
  #task.true <- makeClassifTask(data=data.big,target="category")
  lrn=makeLearner("classif.knn",k=15)
  model <- train(learner=lrn,task=task.train)
  #pred <- predict(newdata=m,model)
  #conf <- calculateConfusionMatrix(pred)
  #err.train[i] <- conf$result[3,3]/(2*n.test)
  pred.test <- predict(newdata=data.test,model)
  conf2 <- calculateConfusionMatrix(pred.test)
  err.test[i] <- conf2$result[3,3]/200
  #pred.true <- predict(newdata=data.big,model)
  #conf3 <- calculateConfusionMatrix(pred.true)
  #err.true[i] <- conf3$result[3,3]/(2*n.true)
}
Err.train <- mean(err.train)
Err.test <- mean(err.test)
Err.true <- mean(err.true)

# boxplot context: K-fold CV carried out 100 times for each 
# value of K shown. Each time it was carried out using a different
# training set of size 40 (so K=40 is LOO).
# Data drawn from same distributions as usual.
# Classifier is 15-NN.
#Err.train = mean(err.train)
#Err.test = mean(err.test)
# Do NOT do this again leave as it is in project PDF
boxplot(h,names=K,xlab="K",ylab="K-fold CV error estimate",
        ylim=c(0.1,0.65))
abline(h=Err.test,col=2,lty=2)
abline(h=Err.train,col=3,lty=2)
abline(h=Err.true,col=4,lty=2)
legend('topright', inset=.05, legend=c("True","Test","Train"), 
       pch=c("_","_","_"), horiz=TRUE, col=c(4,2,3))

#n.test <- 100
#lab1 <- rep(0,n.test)
#lab2 <- rep(1,n.test)
#t <- matrix(0,nrow=z,ncol=1)
#for(i in 1:z){
#    set.seed(i+3)
#    x1 <- mvrnorm(n.test,mu1,sigma)
#    x2 <- mvrnorm(n.test,mu2,sigma)
#    m <- data.frame(obs=matrix(c(x1,x2),nrow=2*n.test,byrow=TRUE),category=as.factor(c(lab1,lab2)))
#    task.train <- makeClassifTask(data=m, target="category")
#    lrn=makeLearner("classif.knn",k=15)
#    model <- train(learner=lrn,task=task.train)
#    pred <- predict(model,newdata=data.big)
#    conf <- calculateConfusionMatrix(pred)
#    t[i] <- conf$result[3,3]/(2*n.true)
#}


K <- c(2:50)
m <- length(K)

errcv <- matrix(0,nrow=m,ncol=2)
Err.true <- matrix(0,nrow=m,ncol=1)

for(i in 1:m){
  lrn = makeLearner("classif.knn",k=15)
  hi = makeResampleDesc("CV",iters=K[i])
  res = resample(lrn, task=task.train, resampling=hi)
  errcv[i,1] <- K[i]
  errcv[i,2] <- res$aggr
}

LOO <- matrix(res$aggr,nrow=m,ncol=1)
true <- matrix(Err.true,nrow=m,ncol=1)
test <- matrix(Err.test,nrow=m,ncol=1)
ave <- matrix(mean(errcv[,2]),nrow=m,ncol=1)
train <- matrix(Err.train,nrow=m,ncol=1)
par(mar=c(rep(4,4)),mfrow=c(1,1))
Error <- structure(c(true,test,train,errcv[,2],ave,LOO), dim=c(m,6), dimnames=list(K,c("True Error","Test Error","Training Error","Cross-Validation Error","Average CV Error","LOO")))
matplot(rownames(Error),Error, type=c('l','l','l','p','l','l'), xlab="Number of folds",ylab="CV-Error",main="CV-errors for varying number of folds",pch=c(2,2,2,"x",2,2),col=c(2,3,9,1,4,6))
legend('bottomright', inset=.05, legend=c("True","Test","Train","CV","Average","LOO"), 
       pch=c("_","_","_","x","_","_"), horiz=TRUE, col=c(2,3,9,1,4,6))

n <- 100
varied <- matrix(0,nrow=n,ncol=5)
for(i in 1:n){
  lrn = makeLearner("classif.knn",k=15)
  hi = makeResampleDesc("CV",iters=5)
  res = resample(lrn, task=task.train, resampling=hi)
  varied[i,1] <- res$aggr
}

for(i in 1:n){
  lrn = makeLearner("classif.knn",k=15)
  hi = makeResampleDesc("CV",iters=10)
  res = resample(lrn, task=task.train, resampling=hi)
  varied[i,2] <- res$aggr
}

for(i in 1:n){
  lrn = makeLearner("classif.knn",k=15)
  hi = makeResampleDesc("CV",iters=50)
  res = resample(lrn, task=task.train, resampling=hi)
  varied[i,3] <- res$aggr
}

for(i in 1:n){
  lrn = makeLearner("classif.knn",k=15)
  hi = makeResampleDesc("CV",iters=n)
  res = resample(lrn, task=task.train, resampling=hi)
  varied[i,4] <- res$aggr
}

lrn = makeLearner("classif.knn",k=15)
hi = makeResampleDesc("LOO")
res = resample(lrn, task=task.train, resampling=hi)
for(i in 1:n){
  varied[i,5] <- res$aggr
}

varied
var(varied)

var.cv <- matrix(0,nrow=5,ncol=1)
for(i in 1:5){
  var.cv[i] <- var(varied)[i,i]
}
var.cv
var.cv1 <- var.cv*10^5
var.cv1

alpha <- 0.95
z <- qnorm(alpha)
h <- n.test
eC <- Err.test
CI.lower <- ((2*h*eC)+(z^2)-(z*sqrt((4*h*eC)+(z^2)-(4*h*(eC)^2))))/
  (2*(h+z^2))
CI.upper <- ((2*h*eC)+(z^2)+(z*sqrt((4*h*eC)+(z^2)-(4*h*(eC)^2))))/
  (2*(h+z^2))
CI.upper
CI <- c(CI.lower,CI.upper)
CI

p <- 50
n.test <- 30
lab1 <- rep(0,n.test)
lab2 <- rep(1,n.test)
mu1 <- c(-0.5,-0.5)
mu2 <- c(0.5,0.5)
sigma <- matrix(c(1,0,0,1),nrow=2,byrow=TRUE)
lrn = makeLearner("classif.knn",k=15)
lrn.lr = makeLearner("classif.logreg")
hi5 = makeResampleDesc("CV",iters=5)
hi10 = makeResampleDesc("CV",iters=10)
hi50 = makeResampleDesc("CV",iters=50)
hi100 = makeResampleDesc("CV",iters=100)
LOO = makeResampleDesc("LOO")
varied.new <- matrix(0,nrow=p,ncol=5)

for(i in 1:p){
  x1 <- mvrnorm(n.test,mu1,sigma)
  x2 <- mvrnorm(n.test,mu2,sigma)
  m <- data.frame(obs=matrix(c(x1,x2),nrow=2*n.test,byrow=TRUE),category=as.factor(c(lab1,lab2)))
  task.train <- makeClassifTask(data=m, target="category")
  res5 = resample(lrn.lr, task=task.train, resampling=hi5)
  res10 = resample(lrn.lr, task=task.train, resampling=hi10)
  res = crossval(lrn.lr,task=task.train,iters=10)
  #res50 = resample(lrn.lr, task=task.train, resampling=hi50)
  #res100 = resample(lrn.lr, task=task.train, resampling=hi100)
  resLOO = resample(lrn.lr, task=task.train, resampling=LOO)
  varied.new[i,1] <- res5$aggr
  varied.new[i,2] <- res10$aggr
  varied.new[i,3] <- res$aggr
  #varied.new[i,4] <- res100$aggr
  varied.new[i,5] <- resLOO$aggr
}
varied.new
var(varied.new)
good.var.cv <- matrix(0,nrow=5,ncol=1)
for(i in 1:5){
  good.var.cv[i] <- var(varied.new)[i,i]
}
good.var.cv
good.var.cv1 <- good.var.cv*10000
good.var.cv1

#model.lr = train(learner=lrn.lr, task=task.train)
#data.pred.lr = predict(object=model.lr,task.true)
#confmat = calculateConfusionMatrix(data.pred.lr)
#Err.true <- confmat$result[3,3]/(2*n.true)
c<- colMeans(varied.new)
acc <- abs(colMeans(varied.new) - Err.true)*100
acc

n <- 10
varied <- matrix(0,nrow=n,ncol=5)
K <- c(5,10,15,50,100)
for(j in 1:length(K)){
  for(i in 1:n){
    res = crossval(lrn.lr,task.train,iters=K[j])
    varied[i,j] <- res$aggr
  }
}
varied
var(varied)*10^6

