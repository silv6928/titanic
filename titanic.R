df <- read.csv("~/titanic/train.csv", sep = ",")

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



