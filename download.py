#import spotipy
#from spotipy.oauth2 import SpotifyClientCredentials
import regex as re
import urllib.request
import csv
from PIL import Image
from io import BytesIO
from os import makedirs
from os.path import join, exists

from utils import print_progress_bar

class cover_art_downloader:

    def __init__(self, titles, artists, image_urls):
        
        
        lim = input("Please enter quantity you want to download (max 65536): ")
        try:
            int(lim)
        except:
            print("Invalid number entered. Set to 10000...\n")
            lim = 10000
        lim = min(65536, lim)
        
        self.folder_path = ".downloaded_images"
        makedirs(self.folder_path, exist_ok=True)
        self.titles = titles[:lim]
        self.artists = artists[:lim]
        self.images = image_urls[:lim]
    
    def file_exists_in_folder(self, filename):
        return exists(join(self.folder_path, filename))
    
    def begin_download(self, size=(256,256)):
        for i in range(len(self.images)):
            print_progress_bar(-1, i, len(self.images))
            filename = f"{self.titles[i][:7].lower()}_{self.artists[i][:7].lower()}.jpeg"
            if not self.file_exists_in_folder(filename):
                try:
                    response = urllib.request.urlopen(self.images[i])
                    img = Image.open(BytesIO(response.read()))
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    img.save(rf"{self.folder_path}/{filename}")
                except:
                    with open("errors.txt", "a") as f:
                        f.write(f"\n{filename}")
        
song_titles = []
song_artists = []
image_urls = []

with open('Music.csv', newline='', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i, row in enumerate(spamreader):
        if i == 0:
            continue  # Skip the first row instead of popping later
        song_titles.append(re.sub(r"[^a-zA-Z0-9]", "", re.sub(r"\s+", "-", row[0])))
        song_artists.append(re.sub(r"[^a-zA-Z0-9]", "", re.sub(r"\s+", "-", row[1])))
        image_urls.append(row[4])

download = cover_art_downloader(song_titles, song_artists, image_urls)
download.begin_download()