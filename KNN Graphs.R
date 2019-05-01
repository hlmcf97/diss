library(mlr)
require(class)
library(klaR)
library(rpart)
set.seed(50)

n1 <- 100
lab1 <- rep(0,n1)
lab2 <- rep(1,n1)
x11 <- rnorm(n1,-0.5,1)
x12 <- rnorm(n1,-0.5,1)
x1 <- cbind(x11,x12)
x21 <- rnorm(n1,0.5,1)
x22 <- rnorm(n1,0.5,1)
x2 <- cbind(x21,x22)
data <- data.frame(obs=matrix(c(x1,x2),byrow=TRUE,nrow=2*n1),category=as.factor(c(lab1,lab2)))
x <- data[,1:2]
g <- data$category
px1 <- seq(-4,4,length.out=69)
px2 <- seq(-4,4,length.out=99)
xnew <- matrix(c(rep(px1,length.out=length(px1)*length(px2)),rep(px2,each=length(px1))),byrow=FALSE,ncol=2)

mod15 <- knn(x, xnew, g, k=1, prob=TRUE)
prob <- attr(mod15, "prob")
prob <- ifelse(mod15=="1", prob, 1-prob)
prob15 <- matrix(prob, length(px1), length(px2))

par(mfrow=c(1,2),mar=c(2,0,2,0),oma=c(rep(0.05,4)))
contour(px1, px2, prob15, levels=0.5, labels="", xlab="", ylab="", main=
          "1-nearest neighbour", axes=FALSE)
points(x, col=ifelse(g==1, "hotpink", "navyblue"))
lines(c(4,-4),c(-4,4),pch=5,col=3)
gd <- expand.grid(x=px1, y=px2)
points(gd, pch=".", cex=1.2, col=ifelse(prob15>0.5, "hotpink", "navyblue"))
box()

mod15 <- knn(x, xnew, g, k=15, prob=TRUE)
prob <- attr(mod15, "prob")
prob <- ifelse(mod15=="1", prob, 1-prob)
prob15 <- matrix(prob, length(px1), length(px2))

contour(px1, px2, prob15, levels=0.5, labels="", xlab="", ylab="", main=
          "15-nearest neighbours", axes=FALSE)
points(x, col=ifelse(g==1, "hotpink", "navyblue"))
lines(c(4,-4),c(-4,4),pch=5,col=3)
gd <- expand.grid(x=px1, y=px2)
points(gd, pch=".", cex=1.2, col=ifelse(prob15>0.5, "hotpink", "navyblue"))
box()

task1 = makeClassifTask(id = "simulation1", data = data, target = "category")
lrn_knn_15 = makeLearner("classif.knn",k=15)
lrn_knn_1 = makeLearner("classif.knn",k=1)
model_knn_15 = train(learner = lrn_knn_15, task = task1)
model_knn_1 = train(learner=lrn_knn_1, task=task1)
data.pred_knn_15 = predict(object=model_knn_15,task1)
data.pred_knn_1 = predict(object=model_knn_1,task1)
calculateConfusionMatrix(data.pred_knn_15)
calculateConfusionMatrix(data.pred_knn_1)

lrn_lr = makeLearner("classif.logreg")
model_lr = train(learner = lrn_lr, task = task1)
data.pred_lr = predict(object=model_lr,task1)
calculateConfusionMatrix(data.pred_lr)


set.seed(51)
n1 <- 100
lab1 <- rep(0,n1)
lab2 <- rep(1,n1)
x11 <- rnorm(n1,-0.5,1)
x12 <- rnorm(n1,-0.5,1)
x1 <- cbind(x11,x12)
x21 <- rnorm(n1,0.5,1)
x22 <- rnorm(n1,0.5,1)
x2 <- cbind(x21,x22)
data <- data.frame(obs=matrix(c(x1,x2),byrow=TRUE,nrow=2*n1),category=as.factor(c(lab1,lab2)))

n2 <- 250
lab1 <- rep(0,n2)
lab2 <- rep(1,n2)
y11 <- rnorm(n2,-0.5,1)
y12 <- rnorm(n2,-0.5,1)
y1 <- cbind(y11,y12)
y21 <- rnorm(n2,0.5,1)
y22 <- rnorm(n2,0.5,1)
y2 <- cbind(y21,y22)
data1 <- data.frame(obs=matrix(c(y1,y2),byrow=TRUE,nrow=2*n2),category=as.factor(c(lab1,lab2)))

n3 <- 10000
lab1 <- rep(0,n3)
lab2 <- rep(1,n3)
y11 <- rnorm(n3,-0.5,1)
y12 <- rnorm(n3,-0.5,1)
y1 <- cbind(y11,y12)
y21 <- rnorm(n3,0.5,1)
y22 <- rnorm(n3,0.5,1)
y2 <- cbind(y21,y22)
data.big <- data.frame(obs=matrix(c(y1,y2),byrow=TRUE,nrow=2*n3),category=as.factor(c(lab1,lab2)))

n4 <- 1000
lab1 <- rep(0,n4)
lab2 <- rep(1,n4)
y11 <- rnorm(n4,-0.5,1)
y12 <- rnorm(n4,-0.5,1)
y1 <- cbind(y11,y12)
y21 <- rnorm(n4,0.5,1)
y22 <- rnorm(n4,0.5,1)
y2 <- cbind(y21,y22)
data2 <- data.frame(obs=matrix(c(y1,y2),byrow=TRUE,nrow=2*n4),category=as.factor(c(lab1,lab2)))

task.train = makeClassifTask(id = "simulation", data = data1, target = "category")
task.test.big = makeClassifTask(id = "simulation2", data = data2, target = "category")
task.test.small = makeClassifTask(id = "pissoff", data = data, target = "category")
task.true = makeClassifTask(id = "trueerror", data=data.big, target="category")

#ho = makeResampleInstance("Holdout",task)
#task.train = subsetTask(task,ho$train.inds[[1]])
#task.test = subsetTask(task,ho$test.inds[[1]])

lrn = makeLearner("classif.knn",k=10)
print(lrn)

model = train(learner = lrn, task = task.train)

data.pred = predict(object=model,task.test)

calculateConfusionMatrix(data.pred)

K <- c(1:50)
m <- length(K)
new <- data1[sample(nrow(data1)),]

ho = makeResampleInstance("Holdout",task.train)
task.train.better = subsetTask(task.train,ho$train.inds[[1]])
task.test.better = subsetTask(task.train,ho$test.inds[[1]])


Err.test.small <- matrix(0,nrow=m,ncol=10)
Err.test.big <- matrix(0,nrow=m,ncol=10)
Err.train <- matrix(0,nrow=m,ncol=10)
Err.true <- matrix(0,nrow=m,ncol=1)

for(i in 1:m){ 
  for(j in 1:10){
    lrn = makeLearner("classif.knn",k=i)
    model = train(learner = lrn, task = task.train)
    data.pred = predict(object=model,task.train)
    confmat <- calculateConfusionMatrix(data.pred)
    Err.train[i,j] <- confmat$result[3,3]/(2*n2)
  }
}

for(i in 1:m){ 
  for(j in 1:10){
    lrn = makeLearner("classif.knn",k=i)
    model = train(learner = lrn, task = task.train)
    data.pred = predict(object=model,task.test.big)
    confmat <- calculateConfusionMatrix(data.pred)
    Err.test.big[i,j] <- confmat$result[3,3]/(2*n4)
  }
}

for(i in 1:m){ 
  for(j in 1:10){
    lrn = makeLearner("classif.knn",k=i)
    model = train(learner = lrn, task = task.train)
    data.pred = predict(object=model,task.test.small)
    confmat <- calculateConfusionMatrix(data.pred)
    Err.test.small[i,j] <- confmat$result[3,3]/(2*n1)
  }
}

#for(i in 1:m){
 # for(j in 1:10){
  #  for(l in 1:10){
   #   task.test = makeClassifTask(id = "simulation2", data = new[(50*j-49):(50*j),], target = "category")
    #  lrn = makeLearner("classif.knn",k=i)
     # model = train(learner = lrn, task = task.train)
      #data.pred = predict(object=model,task.test)
    #  confmat <- calculateConfusionMatrix(data.pred)
     # help[i,10*j-l] <- confmat$result[3,3]/500
  #  }
  #}
#}

#task.train = makeClassifTask(id = "simulation", data = data, target = "category")
#hi = makeResampleDesc("RepCV",folds=10)
#for(i in 1:m){
 # lrn = makeLearner("classif.knn",k=i)
#  res = resample(lrn, task=task.train, resampling=hi)
#  errcv[i,] <- res$aggr
#}

for(i in 1:m){
  lrn = makeLearner("classif.knn",k=i)
  model = train(learner=lrn, task=task.train)
  data.pred = predict(object=model,task.true)
  confmat = calculateConfusionMatrix(data.pred)
  Err.true[i,] = confmat$result[3,3]/(2*n3)
}


#lrn = makeLearner("classif.logreg")
#model = train(learner=lrn, task=task.train)
#data.pred = predict(object=model,task.test.small)
#confmat = calculateConfusionMatrix(data.pred)
#confmat$result[3,3]/(2*n1)


#for(i in 1:m){
 # lrn = makeLearner("classif.knn",k=i)
#  model = train(learner=lrn, task=task.train)
#  data.pred = predict(object=model,task.test)
#  confmat = calculateConfusionMatrix(data.pred)
#  Err.better[i,] = confmat$result[3,3]/500
#}

par(mar=c(rep(4,4)),mfrow=c(1,1))
#err <- c(rowMeans(help))
#err1 <- c(rowMeans(Err1))
err.train <- c(rowMeans(Err.train))
err.test.big <- c(rowMeans(Err.test.big))
err.test.small <- c(rowMeans(Err.test.small))
Error <- structure(c(Err.true,err.test.big,err.test.small,err.train), dim=c(m,4), dimnames=list(K,c("True Error","Test Error (large test set)","Test Error (small test set)","Training Error")))
matplot(rownames(Error),Error, type='l', xlab="K",ylab="Error",main="Errors For Varying K",col=c(1,2,4,6),lwd=c(2,2,2,2))
legend('bottomright', inset=.05, legend=c("True","Test (big)","Test (small)","Training"), 
       pch=c("_","_","_"), horiz=TRUE, col=c(1,2,4,6), cex = 1.3,pt.cex=2)

which.min(Err.true)

tree.sim <- rpart(category~.,data=data,method="class")
tree.sim
par(xpd=TRUE)
plot(tree.sim)
text(tree.sim, use.n=TRUE)

target <- "category"
lrn <- lrn_knn_15
e0 <- makeResampleDesc("Bootstrap",iters=1)

foo <- function(data, indices, lrn, target){
  dt <- data[indices,]
  n <- dim(data)[1]
  P <- matrix(0,nrow=n,ncol=1)
  task <- makeClassifTask(data=dt,target=target)
  task.train <- makeClassifTask(data=data,target=target)
  model <- train(learner=lrn,task=task)
  model.train <- train(learner=lrn,task=task.train)
  ee0 <- resample(lrn,task,resampling=e0)$aggr
  pred <- predict(newdata=dt,model)
  pred.train <- predict(newdata=data,model.train)
  err.bootstrap <- calculateConfusionMatrix(pred)$result[3,3]/n
  eA <- (calculateConfusionMatrix(pred.train)$result[3,3])/n
  R <- (e0-eA)/(gamma-eA)
  rho <- 0.632/(1-(0.368*R))
  err.632 <- 0.368*eA + 0.632*e0
  err.632plus <- (1-rho)*eA + rho*e0
  #print(indices)
  for(i in 1:n){
    P[i] <- sum(indices==i)
  }
  P <- P/n
  summ <- matrix(0,nrow=n,ncol=1)
  #print(P)
  for(i in 1:n){
    pred <- predict(newdata=data[i,],model)
    erroromg <- calculateConfusionMatrix(pred)$result[3,3]
    summ[i] <- ((1/n)-P[i])*erroromg
  }
  #print(summ)
  bias <- sum(summ)
  return(err.632)
}

foo.quant <- function(data, indices, lrn, target){
  tr <- 0.205
  dt<- data[indices,]
  n <- dim(data)[1]
  task <- makeClassifTask(data=dt,target=target)
  model <- train(learner=lrn,task=task)
  pred <- predict(model,newdata=dt)
  e <- (calculateConfusionMatrix(pred))$result[3,3]/n
  quant <- e - tr
  return(quant)
}

lrn_knn_1 <- makeLearner("classif.knn",predict.type="response",k=1)
data.quant <- boot(data=data,statistic=foo.quant,R=11,lrn=lrn_knn_15,target="category")
data.quant$t

boot.data <- boot(data=data,statistic=foo,R=120,lrn=lrn_knn_15, target="category")
boot.data
mean(boot.data$t[,1])

p <- length(which(data$category=="1"))/m

foo_chap_4 <- function(data,indices,lrn,target){
  dt <- data[indices,]
  n <- dim(data)[1]
  task <- makeClassifTask(data=dt,target=target)
  task.train <- makeClassifTask(data=data,target=target)
  model <- train(learner=lrn,task=task)
  model.train <- train(learner=lrn,task=task.train)
  ee0 <- resample(lrn,task,resampling=e0)$aggr
  pred.train <- predict(newdata=data,model.train)
  q <- length(which(pred.train$data[,2]=="1"))/n
  gamma <- p*(1-q)+(1-p)*q
  eA <- (calculateConfusionMatrix(pred.train)$result[3,3])/n
  R <- (ee0-eA)/(gamma-eA)
  rho <- 0.632/(1-(0.368*R))
  err.632 <- 0.368*eA + 0.632*ee0
  err.632plus <- (1-rho)*eA + rho*ee0
  return(c(ee0,err.632,err.632plus))
}
 
boot.again <- boot(data=data,statistic=foo_chap_4,R=200,lrn=lrn_knn_15,target="category")
boot.again
par(mfrow=c(1,1))
hist(boot.again$t[,3])  

