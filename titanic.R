wd <- "C:/Users/Anthony Silva/silvat/titanic/train.csv"
setwd(wd)
df <- read.csv(wd, sep = ",")

#Requirements
install.packages('ggplot2')
library(ggplot2)
install.packages('doBy')
library(doBy)

summary(df)

# Data Preprocessing
df$Survived <- factor(df$Survived)
df$Embarked <- factor(df$Embarked)
df$SibSp <- factor(df$SibSp, levels = seq(0,8,1))
df$SibSp <- ordered(df$SibSp)
df$Parch <- factor(df$Parch, levels = seq(0,6,1))
df$Parch <- ordered(df$Parch)
# Missing Values in Two trips of where they embarked
# Find the people that had cabin's close to the NA's
# Those people are assigned an S, also the most frequent emabarked place
df[grep(pattern = "B2", x = df$Cabin),]
df$Embarked[df$Embarked == ""] <- "S"
df$Embarked <- factor(df$Embarked)


# Missing Age Values
# Trying to estimate using the average age based on if the passenger has Siblings/Spouse Parents/Children
# I.E. If the passengar has multiple sibilings and 2 parents then they are more likely to be younger than those
# That are married with kids. I tried to use to use this logic in order to better fill in the missing values.
df$Age[is.na(df$Age) & df$SibSp > 1 & df$Parch == 2] <- mean(df$Age[!is.na(df$Age) & df$SibSp > 1 & df$Parch == 2])
df$Age[is.na(df$Age) & df$SibSp > 1 & df$Parch == 1] <- mean(df$Age[!is.na(df$Age) & df$SibSp > 1 & df$Parch == 1])
df$Age[is.na(df$Age) & df$SibSp == 0 & df$Parch == 0] <- mean(df$Age[!is.na(df$Age) & df$SibSp == 0 & df$Parch == 0])
# These are the passengers that don't have any parents or children, and are traveling with a sibiling or a spouse.
# The assumption is that these passengers are going older than children since they are traveling without children
# Or they are traveling with their spouse.
df$Age[is.na(df$Age) & df$SibSp > 0 & df$Parch == 0] <- mean(df$Age[!is.na(df$Age) & df$SibSp > 0  & df$Parch == 0])
df$Age[is.na(df$Age) & df$SibSp == 0 & df$Parch == 2] <- mean(df$Age[!is.na(df$Age) & df$SibSp == 0 & df$Parch == 2])
df$Age[grepl('Master', df$Name) & is.na(df$Age) & df$SibSp == 1 & df$Parch == 1] <- mean(df$Age[grepl('Master', df$Name) & !is.na(df$Age) & df$SibSp == 1 & df$Parch == 1])
df$Age[grepl('Miss.', df$Name) & is.na(df$Age) & df$SibSp == 1 & df$Parch == 1] <- mean(df$Age[grepl('Miss.', df$Name) & !is.na(df$Age) & df$SibSp == 1 & df$Parch == 1])
df$Age[grepl('Miss.', df$Name) & is.na(df$Age) & df$SibSp == 1 & df$Parch == 2] <- mean(df$Age[grepl('Miss.', df$Name) & !is.na(df$Age) & df$SibSp == 1 & df$Parch == 2])
df$Age[grepl('Mr.', df$Name) & is.na(df$Age) & df$SibSp == 1 & df$Parch == 2] <- mean(df$Age[grepl('Mr.', df$Name) & !is.na(df$Age) & df$SibSp == 1 & df$Parch == 2])
df$Age[grepl('Mrs.', df$Name) & is.na(df$Age) & df$SibSp == 0 & df$Parch == 1] <- mean(df$Age[grepl('Mrs.', df$Name) & !is.na(df$Age) & df$SibSp == 0 & df$Parch == 1])


