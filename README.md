
# Hi, I'm Ibrahim! 👋 and this is a
Machine Learning Comparison on SMS Spam Detection 

![](https://assets.skyfilabs.com/images/blog/spam-sms-detection.webp)


## Dataset
The Dataset contain 5572 SMS labled with ham (real) and spam
![](https://i.ibb.co/98NQMPB/Screenshot-2021-09-05-234951.png)

The CSV file is included in the repo. Check it out!

### Note: The Dataset is imbalanced by nature
![](https://i.ibb.co/pZ9jJGB/Screenshot-2021-09-05-235403.png)

## 🛠 Install Dependencies
    import numpy as np
    import pandas as pd 
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

## Word Cloud
### The function
```
def visualize(label):
  words=''
  for msg in df[df['Category'] == label]['Message']:
    msg = msg.lower()
    words+=msg + ''
  wordcloud = WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()
```
### Plotting
```
print("Featured words in spam messeges:")
visualize('spam')

print("Featured words in non-spam messeges:")
visualize('ham')
```
![](https://i.ibb.co/X3hN0RG/Screenshot-2021-09-05-235709.png)

## Label Encoding
```
df['b_labels'] = df['Category'].map({'ham':0, 'spam':1})

```
## Feature Extraction
Bag-of-Word (BOW) is used for feature extraction, using the function CountVectorizer in python

## Machine Learning Models
**This a comprehensive between 4 ML models**
- Support Vector Machine
- Decision Tree
- K-Nearest Neighbours
- Random Forest


## Evaluation Metrics
- Confusion Martix
- ROC & AUC 



  
## Authors

- [@96ibman](https://www.github.com/96ibman)

  
## 🚀 About Me
Ibrahim M. Nasser, a Software Engineer, Usability Analyst, 
and a Machine Learning Researcher.


  
## 🔗 Links
[![GS](https://img.shields.io/badge/-Google%20Scholar-blue)](https://scholar.google.com/citations?user=SSCOEdoAAAAJ&hl=en&authuser=2/)

[![linkedin](https://img.shields.io/badge/-Linked%20In-blue)](https://www.linkedin.com/in/ibrahimnasser96/)

[![Kaggle](https://img.shields.io/badge/-Kaggle-blue)](https://www.kaggle.com/ibrahim96/)

  
## Contact

96ibman@gmail.com

  