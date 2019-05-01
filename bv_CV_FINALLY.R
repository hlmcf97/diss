library(mlr)
require(class)
library(klaR)

set.seed(3)
n.test <- 100
lab1 <- rep(0,n.test+3)
lab2 <- rep(1,n.test+3)
x11 <- c(rnorm(n.test,-0.5,1),-2,-2.5,-3)
x12 <- c(rnorm(n.test,-0.5,1),-2,-2.5,-3)
x1 <- cbind(x11,x12)
x21 <- c(rnorm(n.test,0.5,1),2,2.5,3)
x22 <- c(rnorm(n.test,0.5,1),2,2.5,3)
x2 <- cbind(x21,x22)
data.test <- data.frame(obs=matrix(c(x1,x2),byrow=TRUE,nrow=2*n.test+6),category=as.factor(c(lab1,lab2)))

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

n.test2 <- 120
lab1 <- rep(0,n.test2)
lab2 <- rep(1,n.test2)
y11 <- rnorm(n.test2,-0.5,1)
y12 <- rnorm(n.test2,-0.5,1)
y1 <- cbind(y11,y12)
y21 <- rnorm(n.test2,0.5,1)
y22 <- rnorm(n.test2,0.5,1)
y2 <- cbind(y21,y22)
data.test2 <- data.frame(obs=matrix(c(y1,y2),byrow=TRUE,nrow=2*n.test2),category=as.factor(c(lab1,lab2)))

hi5 = makeResampleDesc("CV",iters=5)
hi10 = makeResampleDesc("CV",iters=10)
hi50 = makeResampleDesc("CV",iters=50)
hi100 = makeResampleDesc("CV",iters=100)
hi200 = makeResampleDesc("CV", iters=200)
LOO = makeResampleDesc("LOO")

lrn.lr <- makeLearner("classif.knn",k=15)

task.train <- makeClassifTask(data=data.test, target="category")
model.lr <- train(lrn.lr,task.train)
pred <- predict(model.lr,newdata=data.big)
pred.train <- predict(model.lr,newdata=data.test)
pred.test <- predict(model.lr,newdata=data.test2)
c <- calculateConfusionMatrix(pred)
c.train <- calculateConfusionMatrix(pred.train)
c.test <- calculateConfusionMatrix(pred.test)
err.test <- c.test$result[3,3]/(2*n.test2)
err.train <- c.train$result[3,3]/(2*n.test+6)
err.true <- c$result[3,3]/(2*n.true)


res5 = resample(lrn.lr, task=task.train, resampling=hi5)
res10 = resample(lrn.lr, task=task.train, resampling=hi10)
res50 = resample(lrn.lr, task=task.train, resampling=hi50)
res100 = resample(lrn.lr, task=task.train, resampling=hi100)
res200 = resample(lrn.lr, task=task.train, resampling=hi200)
resLOO = resample(lrn.lr, task=task.train, resampling=LOO)
res5.mat <- matrix(0,nrow=100,ncol=1)
for(i in 0:19){
  r <- resample(lrn.lr, task=task.train, resampling=hi5)
  for(j in 1:5){
    res5.mat[(5*i)+j] <- r$measures.test[j,2]
  }
}
res10.mat <- matrix(0,nrow=100,ncol=1)
for(i in 0:9){
  r <- resample(lrn.lr, task=task.train, resampling=hi10)
  for(j in 1:10){
    res10.mat[(10*i)+j] <- r$measures.test[j,2]
  }
}
res50.mat <- matrix(0,nrow=100,ncol=1)
for(i in 0:1){
  r <- resample(lrn.lr, task=task.train, resampling=hi50)
  for(j in 1:50){
    res50.mat[(50*i)+j] <- r$measures.test[j,2]
  }
}
#res200 = resample(lrn.lr, task=task.train, resampling=hi200)
r5 <- colMeans(res5.mat)
r10 <- colMeans(res10.mat)
r50 <- colMeans(res50.mat)
r100 <- colMeans(res100$measures.test[2])
r <- c(r5,r10,r50,r100)

boxplot(cbind(res5.mat,res10.mat,
              res50.mat,res100$measures.test[2]),
        xlab="K",ylab="CV Error",names=c(5,10,50,100))
abline(h=err.true,col=4,lty=2,lwd=2)
abline(h=err.test+0.01,col=2,lty=2,lwd=2)
abline(h=err.train-0.01,col=3,lty=2,lwd=2)
points(r,col=9,pch="x",lwd=2)
legend('topleft',inset=0.05,legend=c("True","Test","Train"),
       col=c(4,2,3),lty=c(2,2,2),lwd=c(2,2,2),horiz=TRUE,cex = 1.3,pt.cex=1)

# attempt to calculate bias and variance
bias5 <- sum(res5$measures.test[2]-err.true)/5
var5 <- var(res5.mat)
bias10 <- sum(res10$measures.test[2]-err.true)/10
var10 <- var(res10.mat)
bias50 <- sum(res50$measures.test[2]-err.true)/50
var50 <- var(res50.mat)
bias100 <- sum(res100$measures.test[2]-err.true)/100
var100 <- var(res100$measures.test[2]-err.true)
bias200 <- sum(res200$measures.test[2]-err.true)/200
var200 <- var(res200$measures.test[2]-err.true)
biasLOO <- sum(resLOO$measures.test[2]-err.true)/206
varLOO <- var(resLOO$measures.test[2]-err.true)
bias <- c(bias5,bias10,bias50,bias100,bias200,biasLOO)
var <- c(var5,var10,var50,var100,var200,varLOO)
total <- matrix(c(5,10,50,100,200,206,bias,var),nrow=6,ncol=3,byrow=FALSE)
total
total[,2] <- total[,2]*100
total[,3] <- total[,3]*100
