setwd("C:/Users/Lala No.5/Desktop/Final_Thesis")
library(readr)
library(tm)
library(SnowballC)
library(wordcloud)
library(RWeka)
library(quanteda)

df <- read_csv("train_clean.csv")
ind <- sample(1:1482535, 100000, replace = FALSE)
description <- df$item_description
corpus <- Corpus(VectorSource(description))
corpus <- tm_map(corpus, PlainTextDocument)
corpus <- tm_map(corpus, function(x) iconv(enc2utf8(x),
                                           sub = "byte"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removeWords,
                 stopwords('english'))
corpus <- tm_map(corpus, removeWords,
                 c("this", "the", "just", "one", "get", "can"))
corpus <- tm_map(corpus, stemDocument)

# Bigrams
minfreq_bigram<-2

token_delim <- " \\t\\r\\n.!?,;\"()"
bitoken <- NGramTokenizer(corpus,
                          Weka_control(min=2,max=2,
                                       delimiters = token_delim))
two_word <- data.frame(table(bitoken))
sort_two <- two_word[order(two_word$Freq,decreasing=TRUE),]
wordcloud(sort_two$bitoken,sort_two$Freq,random.order=FALSE,
          scale = c(2,0.35),min.freq = minfreq_bigram,
          colors = brewer.pal(8,"Dark2"),max.words=150)


wordcloud(corpus, max.words = 100, random.order = FALSE,
          rot.per = 0.35, use.r.layout = FALSE,
          colors = (brewer.pal(9,"Pastel1")))







# create the corpus object from the item_description column
description <- description[ind]
dcorpus <- corpus(df$item_description)

dcorpus <- tm_map(dcorpus, removeWords, stopwords('english'))
dcorpus <- tm_map(dcorpus, removeWords, c("this", "the", "just", "one", "get", "can"))
dfm2 <- dcorpus %>%
  corpus_sample(size = floor(ndoc(dcorpus) * 0.15)) %>%
  dfm(
    ngrams = 2,
    ignoredFeatures = c("rm", stopwords("english")),
    remove_punct = TRUE,
    remove_numbers = TRUE,
    concatenator = " "
  )


set.seed(456)
textplot_wordcloud(dfm2, min.freq = 2000, random.order = FALSE,
                   rot.per = .25,
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))




