e <- read.csv("final_project_dataset.csv", na.strings = "NaN")

library(ggplot2)

ggplot(e, aes(salary)) + geom_histogram() + facet_wrap(~ poi, ncol=1)
