import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup

page = requests.get('https://scp-wiki.wikidot.com/scp-093')
soup = BeautifulSoup(page.content, 'html.parser')

all_content = soup.find_all('p')
all_data = ''.join([data.text for data in all_content])

st.title('This is a Test App')

st.text('''This will henceforth conain code and certain applications for it 
            to be used later for blogging and portfolio creating purposes.''')