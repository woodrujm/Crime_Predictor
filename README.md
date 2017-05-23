# San Francisco Crime

## Introduction

The motivation for this project stems from the San Francisco public database of police reports. The database consists of information on every logged police report ranging from 2003 to 2017. The information includes things like resolution, category, location, and description. I became interested in predicting the resolution of serious crimes in San Francisco. I defined serious crimes as must felony offenses and, in particular, violent crimes. The reduction of the original two million plus rows can be found in the "clean_data.py" file. Essentially, I decided to reduce the data down to, what I considered, serious crimes and crimes that were not resolved at the scene of the incident.

## Exploratory Data Analysis

I began my exploration of the data by creating a heatmap that takes locations of crimes and outputs them on the GoogleMaps API. This information can be seen in the html and python files named "heatmap." From this visualization it was easily attained that the majority of crimes occurred in the same areas. Furthermore, I was able to find that roughly sixty percent of all crimes go unsolved. Once I had this information I decided to focus on crimes that I found to be more intriguing. In particular, these were crimes that made me feel "unsafe" in a manner of speaking. They were the crimes that I believed to be the most detrimental to a functioning society. To my surprise, after reducing the data, I found that these crimes were actually less likely to have a resolution, something I found to be extremely worrisome.
## Base Model

To get meaningful results I created a base model from a significantly reduced dataset. I essentially removed all text columns and ran Logistic Regression on the remaining information. In particular, the most significant remaining information pertained to the location. The Accuracy, Precision, and Recall from this model were 62.5%, 62%, and .063% respectively. This model was essentially predicting the majority class for each data point, producing results that mirrored the proportions of resolved to non-resolved in the full dataset.
