####Pokemon Winner Prediction#####
####ref: https://www.kaggle.com/jonathanbouchet/pokemon-battles/report
#### https://medium.com/ai-enigma/predicting-pokemon-battle-winner-using-machine-learning-d1ed055ac50
rm(list=ls())

#import libraries
library(data.table)
library(dplyr)
library(caret)
library(randomForest)
library(e1071)

#import dataset
pokemon=fread("C:/Users/nxf51120/Downloads/Pokemon_Battle_Winner_Prediction_using_ML-master/Pokemon_Battle_Winner_Prediction_using_ML-master/datasets/pokemon.csv",sep=",",stringsAsFactors=F)
combats=fread("C:/Users/nxf51120/Downloads/Pokemon_Battle_Winner_Prediction_using_ML-master/Pokemon_Battle_Winner_Prediction_using_ML-master/datasets/combats.csv",sep=",",stringsAsFactors=F)

#display first 10 rows
head(pokemon)

#replace ids with pokemon names
colnames(pokemon)<-c("id","Name","Type.1","Type.2","HP","Attack","Defense","Sp.Atk","Sp.Def","Speed","Generation","Legendary")
# names <- pokemon %>% select(id, Name)
# new_combat_data=combats
# new_combat_data$First_pokemon=names$Name[match(combats$First_pokemon, names$"#")]
# new_combat_data$Second_pokemon=names$Name[match(combats$Second_pokemon, names$"#")]
# new_combat_data$Winner=names$Name[match(combats$Winner, names$"#")]
# head(new_combat_data)

# #Prepare actual Winner column (the actual output for training)    
# combats$Winner[combats$Winner == combats$First_pokemon] = 0
# combats$Winner[combats$Winner == combats$Second_pokemon] = 1

#Taking diff between pokemon stats and normalizing the data
test_combats<-combats
names <- pokemon %>% dplyr::select(id, Name)

test_combats$First_pokemon_name<-sapply(test_combats$First_pokemon, function(x) names$Name[match(x, names$id)])
test_combats$Second_pokemon_name<-sapply(test_combats$Second_pokemon, function(x) names$Name[match(x, names$id)])

test_combats$First_pokemon_attack<-sapply(test_combats$First_pokemon_name, function(x) pokemon$Attack[match(x, pokemon$Name)])
test_combats$Second_pokemon_attack<-sapply(test_combats$Second_pokemon_name, function(x) pokemon$Attack[match(x, pokemon$Name)])
test_combats$Diff_attack<-test_combats$First_pokemon_attack - test_combats$Second_pokemon_attack

test_combats$winner_first_label<-ifelse(test_combats$Winner==test_combats$First_pokemon,'yes','no')

test_combats$First_pokemon_defense<-sapply(test_combats$First_pokemon_name, function(x) pokemon$Defense[match(x, pokemon$Name)])
test_combats$Second_pokemon_defense<-sapply(test_combats$Second_pokemon_name, function(x) pokemon$Defense[match(x, pokemon$Name)])
test_combats$Diff_defense<-test_combats$First_pokemon_defense - test_combats$Second_pokemon_defense

test_combats$First_pokemon_sp_defense<-sapply(test_combats$First_pokemon_name, function(x) pokemon$Sp.Def[match(x, pokemon$Name)])
test_combats$Second_pokemon_sp_defense<-sapply(test_combats$Second_pokemon_name, function(x) pokemon$Sp.Def[match(x, pokemon$Name)])
test_combats$Diff_sp_defense<-test_combats$First_pokemon_sp_defense - test_combats$Second_pokemon_sp_defense

test_combats$First_pokemon_sp_attack<-sapply(test_combats$First_pokemon_name, function(x) pokemon$Sp.Atk[match(x, pokemon$Name)])
test_combats$Second_pokemon_sp_attack<-sapply(test_combats$Second_pokemon_name, function(x) pokemon$Sp.Atk[match(x, pokemon$Name)])
test_combats$Diff_sp_attack<-test_combats$First_pokemon_sp_attack - test_combats$Second_pokemon_sp_attack

test_combats$First_pokemon_speed<-sapply(test_combats$First_pokemon_name, function(x) pokemon$Speed[match(x, pokemon$Name)])
test_combats$Second_pokemon_speed<-sapply(test_combats$Second_pokemon_name, function(x) pokemon$Speed[match(x, pokemon$Name)])
test_combats$Diff_speed<-test_combats$First_pokemon_speed - test_combats$Second_pokemon_speed

test_combats$First_pokemon_HP<-sapply(test_combats$First_pokemon_name, function(x) pokemon$HP[match(x, pokemon$Name)])
test_combats$Second_pokemon_HP<-sapply(test_combats$Second_pokemon_name, function(x) pokemon$HP[match(x, pokemon$Name)])
test_combats$Diff_HP<-test_combats$First_pokemon_HP - test_combats$Second_pokemon_HP

test_combats$First_pokemon_type<-sapply(test_combats$First_pokemon_name, function(x) pokemon$Type.1[match(x, pokemon$Name)])
test_combats$Second_pokemon_type<-sapply(test_combats$Second_pokemon_name, function(x) pokemon$Type.1[match(x, pokemon$Name)])
test_combats$First_pokemon_legendary<-sapply(test_combats$First_pokemon_name, function(x) pokemon$Legendary[match(x, pokemon$Name)])
test_combats$Second_pokemon_legendary<-sapply(test_combats$Second_pokemon_name, function(x) pokemon$Legendary[match(x, pokemon$Name)])

#scale numerical features
temp<- data.frame(test_combats %>% dplyr::select(winner_first_label,Diff_attack ,Diff_defense, Diff_sp_defense,Diff_sp_attack,Diff_speed ,Diff_HP, First_pokemon_legendary, Second_pokemon_legendary))
ind <- sapply(temp, is.numeric)
temp[ind] <- lapply(temp[ind], scale)

#train the model
set.seed(1234)
split <- createDataPartition(y=temp$winner_first_label, p = 0.75, list = FALSE)
train <- temp[split,]
test <- temp[-split,]
regressors<-c('svmLinear','rf','gbm') #'knn','nb','lda',
trControl <- trainControl(method = "cv",number = 5, repeats=3)
timing<-c()
res<-list()
cnt<-0
for(r in regressors){
  cnt<-cnt+1
  start.time <- Sys.time()
  res[[cnt]]<-train(winner_first_label~.,data=train,method=r,trControl = trControl,metric='Accuracy')
  end.time<-Sys.time()
  timing<-c(timing,as.numeric(difftime(end.time,start.time,units="sec")))
}

# cf<- train(winner_first_label~.,data=train, method="rf",trControl = trControl,metric = "Accuracy") #prox=TRUE) #trControl = trControl,

#results on training model
results<-resamples(list("SVM"=res[[1]],"RF"=res[[2]],"gbm"=res[[3]]))
bwplot(results,scales = list(relation = "free"),xlim = list(c(0.5,1), c(0.5,1)))

# accuracy comparison
caret::confusionMatrix(res[[1]])
# > caret::confusionMatrix(res[[1]])
# Cross-Validated (5 fold) Confusion Matrix 
# 
# (entries are percentual average cell counts across resamples)
# 
# Reference
# Prediction   no  yes
# no  48.3  4.2
# yes  4.5 43.0
# 
# Accuracy (average) : 0.9133
# 
# > caret::confusionMatrix(res[[2]])
# Cross-Validated (5 fold) Confusion Matrix 
# 
# (entries are percentual average cell counts across resamples)
# 
# Reference
# Prediction   no  yes
# no  49.7  2.0
# yes  3.1 45.2
# 
# Accuracy (average) : 0.9487
# 
# > caret::confusionMatrix(res[[3]])
# Cross-Validated (5 fold) Confusion Matrix 
# 
# (entries are percentual average cell counts across resamples)
# 
# Reference
# Prediction   no  yes
# no  49.4  2.1
# yes  3.4 45.1
# 
# Accuracy (average) : 0.9457 

# times comparison
names<-c()
for(i in 1:length(res)){
  names[i]<-res[[i]]$method
}
timingData<-data.frame('classifier'=names,'val' = timing)
ggplot(data=timingData,aes(x=reorder(classifier,val),y=val)) + 
  geom_bar(stat='identity') + theme_fivethirtyeight() +
  coord_flip() + 
  xlab('') + ylab('Time [sec]') + ggtitle('Time[sec] spent by classifier')

# Results on Test sample
test_pred <- predict(res[[3]], newdata = test)
confusionMatrix(test_pred, test$winner_first_label)

# > test_pred <- predict(res[[1]], newdata = test)
# > confusionMatrix(test_pred, test$winner_first_label)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   no  yes
# no  6014  504
# yes  585 5396
# 
# Accuracy : 0.9129          
# 95% CI : (0.9078, 0.9178)
# No Information Rate : 0.528           
# P-Value [Acc > NIR] : < 2e-16         
# 
# Kappa : 0.8253          
# Mcnemar's Test P-Value : 0.01534         
# 
# Sensitivity : 0.9114          
# Specificity : 0.9146          
# Pos Pred Value : 0.9227          
# Neg Pred Value : 0.9022          
# Prevalence : 0.5280          
# Detection Rate : 0.4812          
# Detection Prevalence : 0.5215          
# Balanced Accuracy : 0.9130          
# 
# 'Positive' Class : no              
# 
# > # Results on Test sample
# > test_pred <- predict(res[[2]], newdata = test)
# > confusionMatrix(test_pred, test$winner_first_label)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   no  yes
# no  6185  232
# yes  414 5668
# 
# Accuracy : 0.9483          
# 95% CI : (0.9443, 0.9521)
# No Information Rate : 0.528           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8965          
# Mcnemar's Test P-Value : 1.069e-12       
# 
# Sensitivity : 0.9373          
# Specificity : 0.9607          
# Pos Pred Value : 0.9638          
# Neg Pred Value : 0.9319          
# Prevalence : 0.5280          
# Detection Rate : 0.4948          
# Detection Prevalence : 0.5134          
# Balanced Accuracy : 0.9490          
# 
# 'Positive' Class : no              
# 
# > # Results on Test sample
#   > test_pred <- predict(res[[3]], newdata = test)
# > confusionMatrix(test_pred, test$winner_first_label)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   no  yes
# no  6159  256
# yes  440 5644
# 
# Accuracy : 0.9443          
# 95% CI : (0.9402, 0.9483)
# No Information Rate : 0.528           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8885          
# Mcnemar's Test P-Value : 4.017e-12       
# 
# Sensitivity : 0.9333          
# Specificity : 0.9566          
# Pos Pred Value : 0.9601          
# Neg Pred Value : 0.9277          
# Prevalence : 0.5280          
# Detection Rate : 0.4928          
# Detection Prevalence : 0.5132          
# Balanced Accuracy : 0.9450          
# 
# 'Positive' Class : no 

#Decision tree to test the advantage feature
temp<- data.frame(test_combats %>% dplyr::select(winner_first_label,Diff_attack ,Diff_defense, Diff_sp_defense,Diff_sp_attack,Diff_speed ,Diff_HP, First_pokemon_legendary, Second_pokemon_legendary))
ind <- sapply(temp, is.numeric)
temp[ind] <- lapply(temp[ind], scale)

set.seed(2345)
split <- createDataPartition(y=temp$winner_first_label, p = 0.75, list = FALSE)
train <- temp[split,]
test <- temp[-split,]

res.tree<-train(
  winner_first_label~Diff_attack+Diff_defense+Diff_sp_defense+Diff_sp_attack+Diff_speed+Diff_HP+First_pokemon_legendary+Second_pokemon_legendary,
  data=train,method='rpart',
  trControl = trainControl(method = "cv",number = 5, repeats=3))

probs <- predict(res.tree, newdata=test, type='prob')
probs<-data.frame(cbind(probs,winner_first = test$winner_first_label))
probs$winner_first_num<-ifelse(probs$winner_first=='no',0,1)

#Tree Visualization and variables importance
rpart.plot(res.tree$finalModel)
plot(caret::varImp(res.tree))
