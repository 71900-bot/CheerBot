import numpy as np
import pandas as pd
import streamlit as st
import cv2 as cv
import pprint
import webbrowser
import json
import nltk
import random
from fer import FER
from nltk.stem import snowball
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

intent_list = []  # Example: Greeting, Goodbye, ....
train_data = []  # Example: Hello there
train_label = []  # Example: Greeting
responses = {}  # Example: Hi human. What are you thinking now?
list_of_words = []

# Text preprocessing
nltk.download('punkt')
snowballStemmer = snowball.SnowballStemmer("english")


def text_preprocessing(sentence):
    # tokenize the sentences
    tokens = nltk.word_tokenize(sentence)
    # check the word is alphabet or number
    for token in tokens:
        if not token.isalnum():
            tokens.remove(token)
    stem_tokens = []
    for token in tokens:
        stem_tokens.append(snowballStemmer.stem(token.lower()))
    return " ".join(stem_tokens)


# Feature Extraction
vectorizer = CountVectorizer()

# Build NLP Model
clf_dt = DecisionTreeClassifier(random_state=33)


# Generate response
def bot_respond(user_query):  # what user say
    user_query = text_preprocessing(user_query)
    user_query_bow = vectorizer.transform([user_query])
    clf = clf_dt
    predicted = clf.predict(user_query_bow)  # predict the intents
    # When model don't know the intent
    max_proba = max(clf.predict_proba(user_query_bow)[0])
    if max_proba < 0.08 and clf == clf_nb:
        predicted = ['noanswer']
    elif max_proba < 0.3 and not clf == clf_nb:
        predicted = ['noanswer']
    bot_response = ""
    numOfResponses = len(responses[predicted[0]])
    chosenResponse = random.randint(0, numOfResponses-1)
    if predicted[0] == "TimeQuery":
        bot_response = eval(responses[predicted[0]][chosenResponse])
    else:
        bot_response = responses[predicted[0]][chosenResponse]
    return bot_response


def load_model():
    # import training data
    with open("intents.json") as f:
        data = json.load(f)

    # load training data
    for intent in data['intents']:
        for text in intent['text']:
            # Save the data sentences
            preprocessed_text = text_preprocessing(text)
            train_data.append(preprocessed_text)
            # Save the data intent
            train_label.append(intent['intent'])
        intent_list.append(intent['intent'])
        responses[intent['intent']] = intent["responses"]

    # Feature Extraction
    vectorizer.fit(train_data)
    list_of_words = vectorizer.get_feature_names_out()
    train_data_bow = vectorizer.transform(train_data)

    # Train the model
    clf_dt.fit(train_data_bow, train_label)


st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">', unsafe_allow_html=True)

st.sidebar.title('Welcome to CheerBot :)')
mode = st.sidebar.selectbox(
    'Select one', ['Home', 'About Me', 'Depression Test', 'Chat with Me :D', 'Help'])

if mode == "Home":
    def add_bg():
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(https://img.freepik.com/free-vector/abstract-pastel-memphis-patterned-background_53876-98953.jpg?w=2000);
                background-attachment: fixed;
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    add_bg()

    st.markdown("<b><p style='text-align: center; font-size: 60px; font-family: Cursive; color: #ff6aa2;'>CheerBot</p></b>",
                unsafe_allow_html=True)

    st.markdown("<b><p style='text-align: center; font-size: 25px; font-family: papyrus; color: #d780d3;'>Hi my dear! What are you thinking now? :)</p></b>", unsafe_allow_html=True)

    st.markdown('<img style="display: block; margin-left: auto; margin-right: auto; width: 80%;" src="https://og-blog-css.outgrow.co/blog/wp-content/uploads/2019/01/robo_small.gif?x65579"/>', unsafe_allow_html=True)


elif mode == "About Me":

    def add_bg2():
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(https://img.freepik.com/free-vector/abstract-watercolor-background_23-2149059289.jpg?w=2000);
                background-attachment: fixed;
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    add_bg2()

    st.markdown("<b><p style='text-align: center; font-size: 60px; font-family: Cursive; color: #ff655e;'>About Me</p></b>",
                unsafe_allow_html=True)

    st.markdown("<b><p style='text-align: center; font-size: 25px; font-family: papyrus; color: #489741;'>Depict your soul with your colourful smiles everyday :D</p></b>", unsafe_allow_html=True)

    st.video("https://www.youtube.com/watch?v=JjIeWwy7iIA")

    st.markdown("<div style='border: 5px dashed #aad085; padding: 5px; margin: 0px; background-color: #f2f4bc; text-align: center; font-family: Cursive; color: #343a40;'><h1>What is Depression?</h1><p style='font-size:20px'>Depression is a major depressive disorder or dysthymia. It causes people’s feelings of sadness, difficulty in thinking and loss of concentration or interest in activities.</p><p style='font-size:20px'>Frontline healthcare professionals' mental health issues have been identified as a global concern. As a result, occupational groups in the medical field have been recognized as suicide risk population.</p></div>", unsafe_allow_html=True)

    st.markdown("<div style='border: 5px dashed #5dcae5; padding: 5px; margin: 0px; background-color: #ffbdea; text-align: center; font-family: Cursive; color: #343a40;'><h1>How I Function Here ^.^</h1><img style='width: 50%;' src='https://thumbs.gfycat.com/HeftyDescriptiveChimneyswift-size_restricted.gif' /><p style='font-size:20px'>1. Provide an evaluation form that measures your depression level.</p><img style='width: 50%;' src='https://www.insegment.com/blog/wp-content/uploads/2020/11/chatbot-marketing.gif' /><p style='font-size:20px;'>2. Chat with me and you will find motivation in my words of encouragement :)</p><img style='width: 50%;' src='https://www.mnhealthycommunities.org/wp-content/uploads/2019/10/TwoMinnesotans-2.gif' /><p style='font-size:20px'>3. Provide some recommended mental health professionals so that you can get access to them immediately!</p></div>", unsafe_allow_html=True)

elif mode == "Depression Test":

    def add_bg3():
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(https://img.freepik.com/free-vector/rainbow-frame-cute-doodle-border-vector_53876-135969.jpg?w=1380&t=st=1664371564~exp=1664372164~hmac=124501e78f31ffe74eb8a7042e6813c613728e6c2557de2e86da40c41afba1a8);
                background-attachment: fixed;
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    add_bg3()

    st.markdown("<b><p style='text-align: center; margin-bottom: 5px; font-size: 60px; font-family: Cursive; color: #6465b0;'>Let's Do Depression Test!</p></b>",
                unsafe_allow_html=True)

    form = st.button("Online Depression Evaluation Form",
                     help="Click to get directed to the website")

    st.markdown("""---""")

    st.markdown('<img style="display: block; margin-left: auto; margin-right: auto; width: 80%;" src="https://i.pinimg.com/originals/6d/17/24/6d172443e85b3eb3c7f7267e1f162956.gif"/>', unsafe_allow_html=True)

elif mode == "Chat with Me :D":

    def add_bg4():
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(https://cutewallpaper.org/21/pastel-background-gif/Twinkle-star-pixel-art-GIF-Find-on-GIFER.gif);
                background-attachment: fixed;
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    add_bg4()

    st.markdown("<div style='border: 5px dashed #d5c4ff; padding: 5px; margin-bottom: 5px; background-color: #d5fbf4; text-align: center; font-family: Cursive; color: #343a40;'><h1>Talk to Me and CHEER You UP! ^V^</h1></div>", unsafe_allow_html=True)

    with st.spinner('Loading Model...'):
        load_model()

    text = st.text_input('You:')

    if text:
        st.write('CheerBot:')
        with st.spinner('Loading...'):
            st.write(bot_respond(text))

    st.markdown("<img style='display: block; margin-left: auto; margin-right: auto; width: 50%;' src='https://i.pinimg.com/originals/53/09/48/5309488ff206d5bf5850908bdfe78409.gif'/>", unsafe_allow_html=True)

elif mode == "Help":

    def add_bg5():
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(https://static.vecteezy.com/system/resources/previews/004/230/868/non_2x/abstract-hand-drawn-frame-curve-shape-background-in-pastel-color-free-vector.jpg);
                background-attachment: fixed;
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    add_bg5()

    st.markdown("<b><p style='text-align: center; font-size: 50px; font-family: Cursive; color: #b14b27;'>Seek for Mental Health Services?</p></b>",
                unsafe_allow_html=True)

    st.markdown("<img style='display: block; margin-left: auto; margin-right: auto; width: 50%;' src='https://www.mnhealthycommunities.org/wp-content/uploads/2019/10/TwoMinnesotans-2.gif'/>", unsafe_allow_html=True)

    st.markdown("<b><p style='text-align: center; font-size: 25px; font-family: papyrus; color: #e39448;'>Please contact the hotlines below for assistance if you are experiencing mental illness.</p></b>", unsafe_allow_html=True)

    st.markdown("<div style='border: 5px dashed #ff655e; padding: 5px; margin: 0px; background-color: #ffd0c2; font-family: Cursive; color: #d85343;'><ol><li>Aloe Mind<ul><li>Working Hours: Mondays to Saturdays (10.30am to 6pm)</li><li>Contact Number: 017-8038384</li></ul></li><li>Malaysian Mental Health Association (MMHA)<ul><li>Working Hours: Mondays to Fridays (9am to 9pm)</li><li>Contact Number: 017-6133039</li></ul></li><li>Thrive Well (Previously known as SOLS Health)<ul><li>Working Hours: Tuesdays to Saturdays (9am to 6pm)</li><li>Contact Number: 018-900-3247</li></ul></li><li>Mental Illness Awareness and Support Association (MIASA)<ul><li>Working Hours: Mondays to Fridays (9:30am to 6:30pm), Saturdays (9:30am to 1:30pm)</li><li>Contact Number: 03-7932 1409 / 019-236 2423</li></ul></li><li>MentCouch Psychology Centre<ul><li>Working Hours: Mondays to Fridays (9am to 5:30pm), Saturdays (9am to 1:30pm)</li><li>Contact Number: 03-2712 9372</li></ul></li></ol></div>", unsafe_allow_html=True)

    st.markdown("<div style='border: 5px dashed #e8e458; padding: 5px; margin: 0px; background-color: #96d78b; text-align: center; font-family: Cursive; color: #343a40;'><h1>Self-Practice to Cope with Depression</h1><img style='width: 50%;' src='https://i.pinimg.com/originals/10/e6/59/10e6591f0ec9515b71c10af42c3d9d95.gif' /><p style='font-size:20px; margin-top:5px;'>1. Exercise. Every day, go for a 15- to 30-minute brisk walk.</p><img style='width: 50%;' src='https://thehealthturtle.com/wp-content/uploads/2021/04/Anim_StayingHealthy-5a707eb63418c60036e4aa78.gif' /><p style='font-size:20px; margin-top:5px;'>2. Consume nutritious foods and plenty of water. </p><img style='width: 50%;' src='https://1.bp.blogspot.com/-cqkG94rzSk0/XS0xKrZcPPI/AAAAAAAMm-4/wNHPJTfWoW0u9-XTF1jdTdRIYkEGafh1ACLcBGAs/s1600/AS0005506_06.gif' /><p style='font-size:20px; margin-top:5px;'>3. Express yourself. Doing things that stimulate your creativity can be beneficial.</p><img style='width: 50%;' src='https://www.icegif.com/wp-content/uploads/feliz-samado-icegif.gif'/><p style='font-size:20px; margin-top:5px;'>4. Be optimistic. The more good you notice, the more good you will notice.</p></div>", unsafe_allow_html=True)

    st.markdown("<div style='border: 5px dashed #ffff7b; padding: 5px; margin: 0px; background-color: #ffd05c; text-align: center; font-family: Cursive; color: #343a40;'><h1>Words of Motivation ^O^</h1><img style='width: 50%;' src='https://i.pinimg.com/736x/1a/a7/22/1aa722cdd0fc6b20b06d4a3c1f329a78.jpg' /><img style='width: 50%;' src='https://i.pinimg.com/originals/d6/0a/5f/d60a5fca15a9e63faa25017095b29672.png' /><img style='width: 100%;' src='https://i.pinimg.com/originals/da/70/5f/da705f59e7f193a42971c239608a42e4.gif' /><img style='width: 100%;' src='https://i.pinimg.com/originals/29/2a/30/292a30b759e14d71c293b7d1a0b311e9.png'/><img style='width: 100%;' src='https://funvizeo.com/media/memes/16bc100323b92fcc/don-know-what-the-future-holds-and-right-now-thats-okay-memes-3565ddaa2959666c-76de4e99f3620a46.jpg'/><img style='width: 100%;' src='https://i.pinimg.com/originals/cd/5f/cf/cd5fcffb9f73c40edd78ae432c55b7f0.jpg'/></div>", unsafe_allow_html=True)

    st.markdown("<b><p style='text-align: center; font-size: 60px; font-family: Cursive; color: #ff865e;'>Relaxing Music :)</p></b>",
                unsafe_allow_html=True)

    st.video("https://www.youtube.com/watch?v=bP9gMpl1gyQ")
    st.video("https://www.youtube.com/watch?v=l11ZaMYwLOI")
    st.video("https://www.youtube.com/watch?v=fKbNM5sLiCY")
    st.video("https://www.youtube.com/watch?v=-7Sgh60esz4&list=PLjcVduEEXM7MZpQARzlnK5_Xv3NW2Ug1E&index=1")
