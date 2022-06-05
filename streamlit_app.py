import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from nltk.stem.porter import PorterStemmer
import pickle
import streamlit.components.v1 as components

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops=stopwords.words('english')
nltk.download('stopwords')
    
    
st.set_option('deprecation.showPyplotGlobalUse', False)
# app would work faster if you would not read and show the data set
data= pd.read_csv("Suicide_Detection.csv")
numm = data["Unnamed: 0"][len(data)-1]
data.drop("Unnamed: 0", axis=1, inplace = True)

from pathlib import Path

# data in home tab
def home(data):
     # app would work faster if you would not read and show the data set
    if st.checkbox("Show Sample Data"):
        st.table(data.head())
    
    st.header("Go to sidebar for navigation to prediction")

# prediction function    
def predict(data):
    st.header("Classify a sentence")
    text = st.text_area("Enter the Sentence to Check if it is suicidal")

    porter=PorterStemmer()
    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    text = np.array(tokenizer_porter(text))
    
    def remove_stopwords(lower_tokens):
      filtered_words=[]
      for s in lower_tokens:
        temp=[]
        for token in s:
          if token not in stops:
            temp.append(token)
        filtered_words.append(temp)
      return filtered_words

    f_text = np.array(remove_stopwords([text])[0])
    f_text = " ".join(f_text)
    tfidf_vectorizer = pickle.load(open("tfidf.pickle", "rb"))
    ss2 = tfidf_vectorizer.transform([f_text])

    # load the model from disk
    loaded_model = pickle.load(open("lr_model.sav", 'rb'))
    result = loaded_model.predict(ss2)

    if st.button("Predict"):
        if result[0] == 1:
            st.error("This is a Suicidal sentence")
        else:
            st.success("Good news!\n it is not a Suicidal sentence")
    
    if st.checkbox("Need some Example?"):
        st.write("try this statement:")
        st.write(data["text"][7])


if __name__ == "__main__":
    # root = Path(__file__).parents[1]
    st.title("Suicide Predictor")
    st.image("https://cms.qz.com/wp-content/uploads/2018/08/suicide-prediction-animated-final.gif?quality=75&strip=all&w=1200&h=630&crop=1",width = 800)
    nav = st.sidebar.radio("Navigation",["Home","Contribute", "About me", "Contact me"])
    if nav == "Home":
        home(data)

    if nav == "Prediction":
        predict(data)

    if nav == "Contribute":
        st.warning("Contribution to the dataset is not developed yet.. that's why no submit button")
        st.header("Contribute to our dataset")
        text1 = st.text_area("Enter the Sentence")
        label1 = st.selectbox("Select the class",["suicide","non-suicide"],index = 0)
        # if st.button("submit"):
        #     #check if we are not reading the dataset
        #     add_lst = {"Unnamed: 0": [numm+1], "text":[text1],"class":[label1]}
        #     add_lst = pd.DataFrame(add_lst)
        #     add_lst.to_csv(root/"new_Suicide_Detection.csv",mode='a',header = False,index= False)
        #     st.success("Submitted")
    if nav == "About me":
        st.header("About me")
        st.write("I am Akash Maurya, Currently serving in Cognizant as a Programmer Analyst Trainee. I am an IBM-certified Data Scientist. I love playing with data and drawing insights for businesses. I am an expert in Python, Machine Learning, Artificial Intelligence, and Data Science. I have been in this industry for the last 4 years. Currently serving as a part-time instructor and Expert of Machine Learning and Artificial Intelligce.")
        st.write("I am an independent Data Scientist. have experience on varius freelance projects from the field of data science and Machine learning. Deep learning is one of my favorite subject. you can see my achivement at")
        
        st.subheader('View my Achivements')
        components.html("""<div data-iframe-width="150" data-iframe-height="270" data-share-badge-id="c71eb334-9ced-4463-955d-dded93f6f364" data-share-badge-host="https://www.credly.com"></div><script type="text/javascript" async src="//cdn.credly.com/assets/utilities/embed.js"></script>""")
        components.html("""<div data-iframe-width="150" data-iframe-height="270" data-share-badge-id="838f5fcb-b703-46e6-aecc-dec1942a2b58" data-share-badge-host="https://www.credly.com"></div><script type="text/javascript" async src="//cdn.credly.com/assets/utilities/embed.js"></script>""")
      
        
    if nav == "Contact me":
        st.header("Contact me")
