import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('CSV/spam.csv',encoding='latin-1')
dataset.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
dataset.rename(columns={'v1':'target','v2':'text'},inplace=True)

encoder = LabelEncoder()
dataset['target'] = encoder.fit_transform(dataset['target'])

dataset = dataset.drop_duplicates(keep='first')

# print(dataset.head())

# NULL AND DUPLICATE COUNT
# null_vals = dataset.isnull().sum()
# duplicate_vals = dataset.duplicated().sum()
# print("NULL : ",null_vals)
# print("DUPLICATE : ",duplicate_vals)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
dataset['char-num'] = dataset['text'].apply(len)
dataset['word-num'] = dataset['text'].apply(lambda x:len(nltk.word_tokenize(x)))
dataset['sentence-num'] = dataset['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
# print(dataset.head())

# sns.histplot(dataset[dataset['target'] == 0]['word-num'],color='blue')
# sns.histplot(dataset[dataset['target'] == 1]['word-num'],color='red')
# sns.pairplot(data=dataset,hue='target')

df = dataset.drop(columns='text')
# sns.heatmap(df.corr(),annot=True)
# print(df.corr())
# plt.show()

def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    ans = []
    for char in text:
        if(char.isalnum()):
            y.append(char)
    for word in y:
        if(word not in stopwords.words('english')):
            ans.append(word)
    y.clear()

    ps = PorterStemmer()
    for word in ans:
        y.append(ps.stem(word))
    return y


df["transformed_text_array"] = dataset['text'].apply(text_transform)
df['transformed_text_str'] = dataset['text'].apply(text_transform).apply(lambda x:" ".join(x))
# print(df.head())

spam_words = []
for msg in df[df['target'] == 1]['transformed_text_array'].tolist():
    for word in msg:
        spam_words.append(word)

print(len(spam_words))

from collections import Counter

spam_count = pd.DataFrame(Counter(spam_words).most_common(30))

print(spam_count.head())

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# cv = CountVectorizer()
tfv = TfidfVectorizer(max_features=3000)
x = tfv.fit_transform(df['transformed_text_str']).toarray()
y = df['target'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2,random_state=42)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
print(accuracy_score(y_true=y_test,y_pred=y_pred1))
print(precision_score(y_true=y_test,y_pred=y_pred1))


mnb.fit(x_train,y_train)
y_pred2 = mnb.predict(x_test)
print(accuracy_score(y_true=y_test,y_pred=y_pred2))
print(precision_score(y_true=y_test,y_pred=y_pred2))


bnb.fit(x_train,y_train)
y_pred3 = bnb.predict(x_test)
print(accuracy_score(y_true=y_test,y_pred=y_pred3))
print(precision_score(y_true=y_test,y_pred=y_pred3))

import pickle

filename_model = 'Spam_mail_Model.pkl'
filename_vectorizer = 'vectorizer.pkl'
pickle.dump(tfv,open(filename_vectorizer,'wb'))
pickle.dump(mnb,open(filename_model,'wb'))

print("model saved to {filename_model} and {filename_vectorizer}")

# choose the model with the best precision score and accuracy score