# Data preparation and cleaning
# Load packages
setwd("C:/Users/Lala No.5/Desktop/Final_Thesis")
library(readr)
library(ggplot2)
library(tm)

# Read data
df_train <- read_tsv("train.tsv")
df_test <- read_tsv("test.tsv")

# Check NA in data and report
NA_checker <- function(df) {
  col <- colnames(df)
  n <- length(col)
  print(paste("The data has " , dim(df)[1] , " observations."))
  for(i in 1 : n) {
    missed <- length(which(is.na(df[col[i]])))
    print(paste(col[i], " has", missed, " missing values."))
  }
}

# Fill na for description
description_name_merge <- function(df){
  ind <- which(grepl("No description",df$item_description))
  ind <- c(ind,which(is.na(df$item_description)))
  print(paste(length(ind), "missing values found in description,",
              length(ind)/length(df$train_id), "percent."))
  print("Filling those description by item name......")
  df$item_description[ind] <- df$name[ind]

  print("Merging the item name and description for the remaining")
  df$item_description[-ind] <- paste(df$name[-ind],
                                     df$item_description[-ind])
  return(df)
}


#####
# For both brand name and category name, fill na, mark as missing
# take top 300 frequency names and keep as factor column
# mark the remaining as not important brand name or category name
# This is the first general approach, category can be further
# implemented as sentence and pass in to the lstm,
# but the dimension reduction is necessary
#####

# Deal with brand name
brand_cleaning <- function(df){
  brand <- df$brand_name
  brand <- tolower(brand)
  n <- length(brand)
  ind <- which(is.na(brand))


  print(paste("There are", n, "observations,", length(ind) ,
              "of them are missing, which is",
              length(ind)/n, "percent."))
  print("Replacing missing values......")
  print(paste("There are", length(unique(brand)),
              "unique brands."))
  brand[ind] <- "missing brand name"

  brand_tb <- as.data.frame(sort(table(brand), decreasing = T))
  print("The top 10 frequency brands are:")
  print(brand_tb[2:11,])
  print(paste("Except for the missing brand name,
              the top 300 frequency brand name takes",
              sum(brand_tb$Freq[2:301]), "observations"))
  print(paste("Top 300 brand names consist of",
              sum(brand_tb$Freq[2:301])/n,
              "percent of the brand names."))
  print("Transforming the low frequency brand name as small brand .....")
  top300b <- as.character(brand_tb$brand[1:301])
  ind_small <- which(!brand %in% top300b)
  brand[ind_small] <- "small brand"
  print(paste("Now there are", length(unique(brand)),
              "unique brand names ready to be factorized."))
  df$brand_name <- brand
  return(df)
}


# Deal with category name
category_cleaning <- function(df){
  category <- df$category_name
  category <- tolower(category)
  n <- length(category)
  ind <- which(is.na(category))


  print(paste("There are", n, "observations,", length(ind) ,
              "of them are missing, which is",
              length(ind)/n, "percent."))
  print("Replacing missing values......")
  print(paste("There are", length(unique(category)), "unique category."))
  category[ind] <- "missing category name"

  category_tb <- as.data.frame(sort(table(category), decreasing = T))
  print("The top 10 frequency categories are:")
  print(category_tb[2:11,])
  print(paste("the top 300 frequency brand name takes",
              sum(category_tb$Freq[1:150]), "observations"))
  print(paste("Top 300 category names consist of",
              sum(category_tb$Freq[1:150])/n,
              "percent of the brand names."))
  print("Transforming the low frequency category
        name as small category .....")
  top300b <- as.character(category_tb$category[1:150])
  ind_small <- which(!category %in% top300b)
  category[ind_small] <- "small category"
  print(paste("Now there are", length(unique(category)),
              "unique category names ready to be factorized."))
  df$category_name <- category
  return(df)
}

df <- df_train
df <- description_name_merge(df)
df <- brand_cleaning(df)
df <- category_cleaning(df)
df$name <- NULL
df$train_id <- NULL

write_csv(df,"train_clean.csv")





