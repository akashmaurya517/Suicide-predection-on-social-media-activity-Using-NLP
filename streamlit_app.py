import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
st.set_option('deprecation.showPyplotGlobalUse', False)
# app would work faster if you would not read and show the data set
data= pd.read_csv("Suicide_Detection.csv")
numm = data["Unnamed: 0"][len(data)-1]
data.drop("Unnamed: 0", axis=1, inplace = True)

from pathlib import Path

root = Path(__file__).parents[1]

st.title("Suicide Predictor")
st.image("https://cms.qz.com/wp-content/uploads/2018/08/suicide-prediction-animated-final.gif?quality=75&strip=all&w=1200&h=630&crop=1",width = 800)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
if nav == "Home":
    
    # app would work faster if you would not read and show the data set
    if st.checkbox("Show Sample Data"):
        st.table(data.head())
    
    st.header("Go to sidebar for navigation to prediction")
    
if nav == "Prediction":
    st.header("Classify a sentence")
    text = st.text_area("Enter the Sentence to Check if it is suicidal")

    from nltk.stem.porter import PorterStemmer
    porter=PorterStemmer()
    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    text = np.array(tokenizer_porter(text))

    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stops=stopwords.words('english')
    nltk.download('stopwords')


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


    import pickle

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
