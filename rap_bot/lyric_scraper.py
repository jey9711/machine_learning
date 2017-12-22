from urllib.parse import urljoin
from nltk import word_tokenize
from bs4 import BeautifulSoup
import sys
import requests

BASE_URL = "http://genius.com"
artist_url = "http://genius.com/artists/Kanye-west/"

response = requests.get(artist_url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36'})

soup = BeautifulSoup(response.text, "lxml")
for song_link in soup.select('ul.song_list > li > a'):
    link = urljoin(BASE_URL, song_link['href'])
    response = requests.get(link)
    soup = BeautifulSoup(response.text)
    lyrics = soup.find('div', class_='lyrics').text.strip()
    tokens = nltk.word_tokenize(lyrics)
    text = nltk.Text(tokens)
    with open("Output.txt", "w") as text_file:
      print(text, file=text_file)
    # tokenize `lyrics` with nltk
