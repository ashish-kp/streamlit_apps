from bs4 import BeautifulSoup as bs
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
stop = stopwords.words('english') + ['like', 'said', 'would', 'could']
from wordcloud import WordCloud
import streamlit as st

# Unnecessary imports, I know...

from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import sys

headers = {
    'User-Agent': 'My User Agent 1.0',
    'From': 'youremail@domain.com'
}

st.title('Erotic Stories WordCloud Generator')

st.text('''For the entered keyword, the relevantly 
    tagged stories in Literotica are scraped.''')
st.text('And a WordCloud is generated from the text.')
st.text('Can be used for story validation.')

inp = st.text_input('Enter a keyword')

if inp != None:

    st.text('Please wait... The process takes some time.')

    page = requests.get('https://tags.literotica.com/' + inp + '/', headers = headers)
    soup = bs(page.content, 'html.parser')

    content = soup.find('div', class_ = 'L_gH')
    
    if content != None:
        all_links = content.select('a')

        store_links = []

        count = 3
        for link in all_links:
            if count % 4 == 0:
                store_links.append(link['href'])
            count += 1

        st.text('Links are retrieved. Now the text...')

        stories = []

        for link in store_links:
            page = requests.get(link, headers = headers)
            soup = bs(page.content, 'html.parser')
            story = soup.select('div.panel.article.aa_eQ')
            stories.append(story[0].text)

        all = ''.join(stories)

        st.text('Here\'s the WordCloud:')

        wordcloud = WordCloud(stopwords = stop,background_color = 'white',width=2500,height=2000).generate(all)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis('off')

        st.pyplot(fig)
