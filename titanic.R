wd <- paste(getwd(),"/titanic/train.csv", sep ="")
df <- read.csv(wd, sep = ",")

#Requirements
install.packages('ggplot2')
library(ggplot2)

summary(df)
# Data Preprocessing
df$Survived <- factor(df$Survived)
df$Embarked <- factor(df$Embarked)
df$SibSp <- factor(df$SibSp, levels = seq(0,8,1))
df$Parch <- factor(df$Parch, levels = seq(0,6,1))
# Missing Values in Two trips of where they embarked
# Find the people that had cabin's close to the NA's
# Those people are assigned an S, also the most frequent emabarked place
df[grep(pattern = "B2", x = df$Cabin),]
df$Embarked[df$Embarked == ""] <- "S"
df$Embarked <- factor(df$Embarked)



# Data Exploration and Insights
hist(df$Age[df$Sex == 'female'])
hist(df$Age[df$Sex == 'male'])

