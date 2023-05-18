import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/Users/vridhijain/SDM-final project/Coursera.csv")
data.shape #3522 courses and 7 columns with different attributes
data.isnull().sum()
data = data[['Course Name','Difficulty Level','Course Description','Skills', 'Course Rating']]

##Data preprocessing

# Removing spaces between the words (Lambda funtions can be used as well)

data['Course Name'] = data['Course Name'].str.replace(' ',',')
data['Course Name'] = data['Course Name'].str.replace(',,',',')
data['Course Name'] = data['Course Name'].str.replace(':','')
data['Course Description'] = data['Course Description'].str.replace(' ',',')
data['Course Description'] = data['Course Description'].str.replace(',,',',')
data['Course Description'] = data['Course Description'].str.replace('_','')
data['Course Description'] = data['Course Description'].str.replace(':','')
data['Course Description'] = data['Course Description'].str.replace('(','')
data['Course Description'] = data['Course Description'].str.replace(')','')

#removing paranthesis from skills columns 
data['Skills'] = data['Skills'].str.replace('(','')
data['Skills'] = data['Skills'].str.replace(')','')

##Data Processing
data['tags'] = data['Course Name'] +' '+ data['Difficulty Level'] +' '+ data['Course Description'] +' '+ data['Skills']+' '+ data['Course Rating']
new_df = data[['Course Name','tags']]
new_df['tags'] = data['tags'].str.replace(',',' ')
new_df['Course Name'] = data['Course Name'].str.replace(',',' ')
new_df.rename(columns = {'Course Name':'course_name'}, inplace = True)
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower()) #lower casing the tags column

##Vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

##Stemming

import nltk #for stemming process
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
#defining the stemming function
def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)
  
new_df['tags'] = new_df['tags'].apply(stem) #applying stemming on the tags column

##Similarity Measure
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

##Recommendation System

def recommend(course):
    course_index = new_df[new_df['course_name'] == course].index[0]
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:7]
    
    for i in course_list:
        print(new_df.iloc[i[0]].course_name)
        
recommend('Business Strategy Business Model Canvas Analysis with Miro') 

