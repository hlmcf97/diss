library(mlr)
library(BBmisc)
library(kknn)
library(dplyr)
library(boot)
library(ipred)
library(rpart)
library(rsample)

weather <- read.csv("~/Downloads/weatherAUS.csv",header=TRUE)
weather$Month <- as.factor(weather$Month)
weather <- weather[,c("Month","Location","MinTemp","MaxTemp",
                      "Rainfall","WindGustDir","WindGustSpeed",
                      "WindDir9am","WindDir3pm","WindSpeed9am",
                      "WindSpeed3pm","Humidity9am","Humidity3pm",
                      "Pressure9am","Pressure3pm","Temp9am",      
                      "Temp3pm","RainToday","RainTomorrow")]
weather <- na.omit(weather)
weather$Location <- droplevels(weather$Location)

stratified <- function(df, group, size, select = NULL, 
                       replace = FALSE, bothSets = FALSE) {
  if (is.null(select)) {
    df <- df
  } else {
    if (is.null(names(select))) stop("'select' must be a named list")
    if (!all(names(select) %in% names(df)))
      stop("Please verify your 'select' argument")
    temp <- sapply(names(select),
                   function(x) df[[x]] %in% select[[x]])
    df <- df[rowSums(temp) == length(select), ]
  }
  df.interaction <- interaction(df[group], drop = TRUE)
  df.table <- table(df.interaction)
  df.split <- split(df, df.interaction)
  if (length(size) > 1) {
    if (length(size) != length(df.split))
      stop("Number of groups is ", length(df.split),
           " but number of sizes supplied is ", length(size))
    if (is.null(names(size))) {
      n <- setNames(size, names(df.split))
      message(sQuote("size"), " vector entered as:\n\nsize = structure(c(",
              paste(n, collapse = ", "), "),\n.Names = c(",
              paste(shQuote(names(n)), collapse = ", "), ")) \n\n")
    } else {
      ifelse(all(names(size) %in% names(df.split)),
             n <- size[names(df.split)],
             stop("Named vector supplied with names ",
                  paste(names(size), collapse = ", "),
                  "\n but the names for the group levels are ",
                  paste(names(df.split), collapse = ", ")))
    }
  } else if (size < 1) {
    n <- round(df.table * size, digits = 0)
  } else if (size >= 1) {
    if (all(df.table >= size) || isTRUE(replace)) {
      n <- setNames(rep(size, length.out = length(df.split)),
                    names(df.split))
    } else {
      message(
        "Some groups\n---",
        paste(names(df.table[df.table < size]), collapse = ", "),
        "---\ncontain fewer observations",
        " than desired number of samples.\n",
        "All observations have been returned from those groups.")
      n <- c(sapply(df.table[df.table >= size], function(x) x = size),
             df.table[df.table < size])
    }
  }
  temp <- lapply(
    names(df.split),
    function(x) df.split[[x]][sample(df.table[x],
                                     n[x], replace = replace), ])
  set1 <- do.call("rbind", temp)
  
  if (isTRUE(bothSets)) {
    set2 <- df[!rownames(df) %in% rownames(set1), ]
    list(SET1 = set1, SET2 = set2)
  } else {
    set1
  }
}

lrn.knn <- makeLearner("classif.kknn",predict.type="response",k=15)
lrn.lr <- makeLearner("classif.logreg",predict.type="response")
lrn.dt <- makeLearner("classif.rpart",predict.type="response")

set.seed(1234)
weat <- stratified(weather,c("Location","RainTomorrow"),0.004)
task.weat <- makeClassifTask(data=weat,target="RainTomorrow",positive="Yes")
m <- dim(weat)[1]

weat.tree <- rpart(RainTomorrow~., data=weat, method="class",
                   parms=list(split="information"))

weat.knn <- train(lrn.knn,task.weat)
weat.lr <- train(lrn.lr,task.weat)
weat.dt <- train(lrn.dt,task.weat)

e.train.knn <- (calculateConfusionMatrix(
  predict(weat.knn,newdata=weat)))$result[3,3]/m
e.train.lr <- (calculateConfusionMatrix(
  predict(weat.lr,newdata=weat)))$result[3,3]/m
e.train.dt<- (calculateConfusionMatrix(
  predict(weat.dt,newdata=weat)))$result[3,3]/m

split <- initial_split(weat, prop=2/3)
weat.train <- training(split)
weat.test <- testing(split)
m1 <- dim(weat.test)[1]
task.train.weat <- makeClassifTask(data=weat.train,
                                   target="RainTomorrow")
task.test.weat <- makeClassifTask(data=weat.test, 
                                  target="RainTomorrow")
weat.train.knn <- train(lrn.knn,task.train.weat)
weat.train.lr <- train(lrn.lr,task.train.weat)
weat.train.dt <- train(lrn.dt,task.train.weat)
e.test.knn <- (calculateConfusionMatrix(
  predict(weat.train.knn,newdata=weat.test)))$result[3,3]/m1
e.test.lr <- (calculateConfusionMatrix(
  predict(weat.train.lr,newdata=weat.test)))$result[3,3]/m1
e.test.dt <- (calculateConfusionMatrix(
  predict(weat.train.dt,newdata=weat.test)))$result[3,3]/m1
err.test <- c(e.test.knn,e.test.lr,e.test.dt)

h <- m1
z <- qnorm(0.975)
eC <- err.test
d <- 0.05

lwr <- ((2*h*eC)+(z^2)-(z*sqrt((4*h*eC)+(z^2)-(4*h*(eC^2)))))/(2*(h+(z^2)))
upr <- ((2*h*eC)+(z^2)+(z*sqrt((4*h*eC)+(z^2)-(4*h*(eC^2)))))/(2*(h+(z^2)))
ci1 <- matrix(c(lwr,upr),nrow=3,ncol=2,byrow=FALSE)

lwr1 <- eC - sqrt((1/h)*log(2/d))
upr1 <- eC + sqrt((1/h)*log(2/d))
ci2 <- matrix(c(lwr1,upr1),nrow=3,ncol=2,byrow=FALSE)

n.all <- dim(weather)[1]
e.true.knn <- (calculateConfusionMatrix(
  predict(weat.knn,newdata=weather)))$result[3,3]/n.all
e.true.lr <- (calculateConfusionMatrix(
  predict(weat.lr,newdata=weather)))$result[3,3]/n.all
e.true.dt <- (calculateConfusionMatrix(
  predict(weat.dt,newdata=weather)))$result[3,3]/n.all

cv5 <- makeResampleDesc(method="CV", iters=5)
cv10 <- makeResampleDesc(method="CV", iters=10)  
LOO <- makeResampleDesc(method="LOO")

knn.cv5 <- resample(lrn.knn,task.weat,resampling=cv5)
t.knn.cv5 <- knn.cv5$runtime
e.knn.cv5 <- knn.cv5$aggr
lr.cv5 <- resample(lrn.lr,task.weat,resampling=cv5)
t.lr.cv5 <- lr.cv5$runtime
e.lr.cv5 <- lr.cv5$aggr
dt.cv5 <- resample(lrn.dt,task.weat,resampling=cv5)
t.dt.cv5 <- dt.cv5$runtime
e.dt.cv5 <- dt.cv5$aggr

knn.cv10 <- resample(lrn.knn,task.weat,resampling=cv10)
t.knn.cv10 <- knn.cv10$runtime
e.knn.cv10 <- knn.cv10$aggr
lr.cv10 <- resample(lrn.lr,task.weat,resampling=cv10)
t.lr.cv10 <- lr.cv10$runtime
e.lr.cv10 <- lr.cv10$aggr
dt.cv10 <- resample(lrn.dt,task.weat,resampling=cv10)
t.dt.cv10 <- dt.cv10$runtime
e.dt.cv10 <- dt.cv10$aggr

knn.loo <- resample(lrn.knn,task.weat,resampling=LOO)
t.knn.loo <- knn.loo$runtime
e.knn.loo <- knn.loo$aggr
lr.loo <- resample(lrn.lr,task.weat,resampling=LOO)
t.lr.loo <- lr.loo$runtime
e.lr.loo <- lr.loo$aggr
dt.loo <- resample(lrn.dt,task.weat,resampling=LOO)
t.dt.loo <- dt.loo$runtime
e.dt.loo <- dt.loo$aggr

scv5 <- makeResampleDesc(method="CV", iters=5, stratify = TRUE)
scv10 <- makeResampleDesc(method="CV", iters=10, stratify=TRUE)

knn.scv5 <- resample(lrn.knn,task.weat,resampling=scv5)
t.knn.scv5 <- knn.scv5$runtime
e.knn.scv5 <- knn.scv5$aggr
lr.scv5 <- resample(lrn.lr,task.weat,resampling=scv5)
t.lr.scv5 <- lr.scv5$runtime
e.lr.scv5 <- lr.scv5$aggr
dt.scv5 <- resample(lrn.dt,task.weat,resampling=scv5)
t.dt.scv5 <- dt.scv5$runtime
e.dt.scv5 <- dt.scv5$aggr

knn.scv10 <- resample(lrn.knn,task.weat,resampling=scv10)
t.knn.scv10 <- knn.scv10$runtime
e.knn.scv10 <- knn.scv10$aggr
lr.scv10 <- resample(lrn.lr,task.weat,resampling=scv10)
t.lr.scv10 <- lr.scv10$runtime
e.lr.scv10 <- lr.scv10$aggr
dt.scv10 <- resample(lrn.dt,task.weat,resampling=scv10)
t.dt.scv10 <- dt.scv10$runtime
e.dt.scv10 <- dt.scv10$aggr

K <- matrix(0,nrow=30,ncol=1)
for(i in 1:30){
  K[i] <- i
}

p <- 2

true.knn <- matrix(0,nrow=length(K),ncol=1)
for(i in 1:length(K)){
  lrn.knns <- makeLearner("classif.kknn",k=K[i])
  weat.knns <- train(learner=lrn.knns,task=task.weat)
  knn.trues <- predict(weat.knns,newdata=weather)
  e.true.knns <- (calculateConfusionMatrix(knn.trues))$result[3,3]/n.all
  true.knn[i] <- e.true.knns
}

cv10.knn <- matrix(0,nrow=length(K),ncol=p)
for(i in 1:length(K)){
  lrn.knnt <- makeLearner("classif.kknn",k=K[i])
  weat.knnt <- train(learner=lrn.knnt,task=task.weat)
  for(j in 1:p){
    knn.cv10t <- resample(lrn.knnt,task.weat,resampling=cv10)$aggr
    cv10.knn[i,j] <- knn.cv10t
  }
}

cv10.knn1 <- rowMeans(cv10.knn)

plot(true.knn,xlab="K",ylab="Error",ylim=c(0.16,0.35),lwd=2)
points(cv10.knn1,col=4,lwd=2)
legend("topright",legend=c("True","10-fold CV"),col=c(1,4),inset=0.05,pch=c("_","_"),
       pt.cex=2,cex=1.1)

e0 <- makeResampleDesc("Bootstrap",iters=1)
knn.e0 <- resample(lrn.knn,task.weat,resampling=e0)$aggr
p <- length(which(weat$RainTomorrow=="Yes"))/m
q <- length(which(knn.train$data[,2]=="Yes"))/m
gamma <- p*(1-q)+(1-p)*q

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
  R <- (ee0-eA)/(gamma-eA)
  rho <- 0.632/(1-(0.368*R))
  err.632 <- 0.368*eA + 0.632*ee0
  err.632plus <- (1-rho)*eA + rho*ee0
  for(i in 1:n){
    P[i] <- sum(indices==i)
  }
  P <- P/n
  summ <- matrix(0,nrow=n,ncol=1)
  for(i in 1:n){
    pred <- predict(newdata=data[i,],model)
    erroromg <- calculateConfusionMatrix(pred)$result[3,3]
    summ[i] <- ((1/n)-P[i])*erroromg
  }
  bias <- sum(summ)
  return(bias,err.bootstrap,ee0,err.632,err.632plus,1)
}

knn.boot <- boot(weat,foo,R=100,lrn=lrn.knn,target="RainTomorrow")
dt.boot <- boot(weat,foo,R=100,lrn=lrn.dt,target="RainTomorrow",
                strata=weat$Location)

hist(knn.boot$t[,2],xlab="Bootstrap Estimate",main="")
abline(v=c[1,1],lty=2)
abline(v=c[1,2],lty=2)
abline(v=e.true.knn,lty=2,col=2)

hist(knn.boot$t[,3],xlab="E0 Estimate",main="")
abline(v=c[2,1],lty=2)
abline(v=c[2,2],lty=2)
abline(v=e.true.knn,lty=2,col=2)

hist(knn.boot$t[,4],xlab=".632 Estimate",main="")
abline(v=c[3,1],lty=2)
abline(v=c[3,2],lty=2)
abline(v=e.true.knn,lty=2,col=2)

hist(knn.boot$t[,5],xlab=".632+ Estimate",main="")
abline(v=c[4,1],lty=2)
abline(v=c[4,2],lty=2)
abline(v=e.true.knn,lty=2,col=2)

boot.ci(dt.boot,conf=0.95,type=c("stud","perc"),index=c(4,6))

weat_for_lr <- stratified(weather,c("Location","RainTomorrow"),0.01)
task.weat_for_lr <- makeClassifTask(data=weat_for_lr,target="RainTomorrow",positive="Yes")
m2 <- dim(weat_for_lr)[1]
weat.lr.bigger <- train(lrn.lr,task.weat_for_lr)
e.train.lr.bigger <- (calculateConfusionMatrix(
  predict(weat.lr.bigger,newdata=weat_for_lr)))$result[3,3]/m2
e.true.lr.bigger <- (calculateConfusionMatrix(
  predict(weat.lr.bigger,newdata=weather)))$result[3,3]/dim(weather)[1]

lr.boot <- boot(weat_for_lr,foo,R=100,lrn=lrn.lr,
                target="RainTomorrow",strata=weat_for_lr$Location)


























