### required packages
library("caret")
library("plyr")
library("reshape")
library("knitr")
library("dplyr")
library("tidyverse")
getwd()
### import and check data 
cenus <- read.csv("adult.csv")
str(cenus)
head(cenus)
dim(cenus)
cenus %>% group_by(income) %>% summarise(Number = length(income)) 

### NAs and empty values
cenus <- read.csv("adult.csv", na.strings = c("?",""))
cenus[!complete.cases(cenus),]
sum(!complete.cases(cenus))
 sapply(cenus, FUN = function(l) {
  table(is.na(l))
})
table(which(is.na(cenus['workclass'])) %in% which(is.na(cenus['occupation'])))
cenus <- na.omit(cenus)
sum(!complete.cases(cenus))

### check data class before applying machine learning 
sapply(cenus, class) %>% knitr::kable()
cenus$age <- as.numeric(cenus$age)
cenus$hours.per.week <- as.numeric(cenus$hours.per.week)

### check variables effect on income
#--workclass--#
workclass_variable <- cenus%>% filter(!is.na(workclass)) %>% group_by(workclass) %>% 
  summarise( morethan50 = sum(income == ">50K")/ n()*100, lessthan50 = sum(income == "<=50K")/n()*100) %>%  as.data.frame()
workclass_variable <- reshape::melt(workclass_variable, id= "workclass") 

ggplot(workclass_variable, aes(x=workclass, y= value, fill = variable, label = round(value,2))) + 
  geom_bar(stat="identity") +
  geom_text(position = position_stack(vjust=0.5))+
  labs(x= "workclass", y = "Percentage", title = "Relation between workclass and income")
#--age--#
ggplot(cenus, aes(x=income, y = age, color = income, fill = income)) +
  geom_violin () +
  labs( title = "People who are older earn more",
        subtitle = "Age")
#--education--#
ggplot(cenus, aes(x=education, color = income, fill = income)) +
  geom_bar(position = "fill")+
  coord_polar() +
  labs( title = "People with higher education earn more",
        subtitle = "Education")
#--marital.status--#
ggplot(cenus, aes(x=marital.status, color = income, fill = income)) +
  geom_bar(position = "fill")+
  coord_flip() +
  labs( title = "People married likley to earn more",
        subtitle = "marital.status")
#--occupation--#
occupation_variable <- as.data.frame(prop.table(table(cenus$occupation, cenus$income), 1) * 100) %>% mutate(Freq = round(Freq,2))
treemap::treemap(occupation_variable, c("Var1", "Var2","Freq"), vSize = "Freq", type="index")
ggplot(occupation_variable, aes(x=Var1, y= Freq, fill = Var2, label = Freq)) + 
  geom_bar(stat="identity") +
  coord_flip()+
  geom_text(position = position_stack(vjust=0.5))+
  labs(x= "occupation", y = "Percentage", title = "Relation between Occupation and income")

#--relationship--#
ggplot(cenus, aes(x=relationship, color = income, fill = income)) +
  geom_bar(position = "fill")+
  coord_flip() +
  labs( title = "people in families earn more ",
        subtitle = "Relationship")
#--Race--#
ggplot(cenus, aes(x=race, color = income, fill = income)) +
  geom_bar(position = "fill")+
  labs( title = "income differs based on the race ",
        subtitle = "race")
#--sex--#
sex_var <- cenus %>% group_by(sex) %>% summarise(total = sum(income == ">50K")/n()*100)
pie(sex_var$total, labels = sex_var$sex, main = "Male earn >50k more than Female", col = sex_var$sex)
box()
#--hours.per.week--##
ggplot(cenus, aes(x = income, y = hours.per.week, fill = income)) +
  geom_boxplot(outlier.shape = NA ) +
  labs(x = "Income", y = "Hours per Week", title = "People who work more hours earn more",
       subtitle = "hours.per.week")
#--native.country--##
ggplot(cenus, aes(x = native.country, fill = income, color = income)) +
  geom_bar( width = 0.8, position = "fill") +
  coord_flip() +
  labs(x = "Native Country", y = "Percent", title = " native country")



#--pre-processing data and remove unusfull predictors--##
ggplot(cenus, aes(x=fnlwgt, y= income, fill=income )) +geom_area()
qplot(cenus$income,cenus$fnlwgt )
dat_train<- select(cenus,-fnlwgt)


# Now we will apply the machine learning algorithm
# first divide the data to training and test sets.
index <- createDataPartition(dat_train$income, times = 1, p= 0.70, list = FALSE)
train_set <- dat_train %>% slice(index)
test_set <- dat_train %>% slice(-index)
dim(train_set)
dim(test_set)
#--fit a decession tree for categorical outcome ,since we have more than two variables--##
tree.model <- train(income ~ . ,
                    data = train_set,method = "rpart")
plot(tree.model)
y_hat_tree <- predict(tree.model,test_set)
confusionMatrix(y_hat_tree,reference = test_set$income)$overall["Accuracy"]

tree.model_2 <- train(income ~ . ,
                      data = train_set,method = "rpart",
                      tuneGrid=data.frame(cp=seq(0,0.05,len=25)))
plot(tree.model_2)
y_hat_tree_2 <- predict(tree.model_2,test_set)
confusionMatrix(y_hat_tree_2,reference = test_set$income)$overall["Accuracy"]

##KNN module##
control <- trainControl(method = "cv",number= 10, p=0.9)
knn_model_fit <- train(income~. , data= train_set, method= "knn", tuneGrid = data.frame(k=seq(5,25,2)),truControl= control)
plot(knn_model_fit)
model_fitbest <- knn3(income ~ . , data = train_set , k= 17)
y_hat_knn <- predict(model_fitbest,test_set,type  ="class")
confusionMatrix(y_hat_knn,test_set$income)$overall["Accuracy"]

##--Random forest method--##
set.seed(1)
library(randomForest)
control <- trainControl( method = "cv", number = 5, p= .8)
grid <- expand.grid(minNode=c(1,2,3,4,5),predFixed=c(10,15,25,35,50))
ff <- train(income ~., method = "Rborist",data=train_set,nTree=50,trControl= control, tuneGrid=grid, nSamp=5000 )
ff$bestTune
ranfor.model <- randomForest::randomForest(income ~ .
                                           , data = train_set, trees=1000, minNode=1, predFixed=10  )

y_hat_forest<- predict(ranfor.model,test_set)
confusionMatrix(y_hat_forest, test_set$income)$overall["Accuracy"]
imp <- importance(ranfor.model)
## adjust the module by removing the least effective variable
adjus_ranfor.model <- randomForest::randomForest(income ~ .-native.country
                                           , data = train_set, trees=1000, minNode=1, predFixed=10  )
y_hat_forestad<- predict(adjus_ranfor.model,test_set)
confusionMatrix(y_hat_forestad, test_set$income)$overall["Accuracy"]

imp <- importance(adjus_ranfor.model)
imp %>% kable()


