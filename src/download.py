import regex as re
import csv
from PIL import Image
from yt_dlp import YoutubeDL
import eyed3

from urllib import request as rq
import urllib.request

from io import BytesIO
from os import makedirs, rename, remove
from os.path import join, exists

from utils import print_progress_bar
from spectrogram import generate_spectrogram

class cover_art_downloader:

    def __init__(self, titles, artists, image_urls, folder_path=".downloaded_iamges"):
        
        
        lim = input("Please enter quantity you want to download (max 65536): ")
        try:
            lim = int(lim)
        except:
            print("Invalid number entered. Set to 10000...\n")
            lim = 10000
        lim = min(65536, lim)
        
        self.folder_path = folder_path
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
 
def download_entire_file():       
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


def download_single_image(image_url):
    download = cover_art_downloader(["TEST_IMAGE"], ["2_TEST_TITLE"], [image_url], ".test_images")
    download.begin_download()
    
    
    
    
class audio_downloader():
    
    def __init__(self, folder_path="data/temp_song_archive/"):
        
        if not exists(folder_path):
            makedirs(folder_path)
        
        if not exists("data/spectrograms/"):
            makedirs("data/spectrograms/")
        
        self.data = "data/Music.csv"
        
        lim = input("Please enter quantity you want to download (max 65536): ")
        try:
            lim = int(lim)
        except:
            print("Invalid number entered. Set to 10...\n")
            lim = 10
        self.lim = min(65536, lim)
        
        self.folder_path = folder_path
        makedirs(self.folder_path, exist_ok=True)
        
        self.extract_data()
        self.download_tracks()
        
        
    def extract_data(self):
        self.songs = []

        with open(self.data, newline='', encoding="utf8") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(spamreader):
                if i == 0:
                    continue
                self.songs.append([re.sub(r"[^a-zA-Z0-9]", "", re.sub(r"\s+", "-", row[0])), re.sub(r"[^a-zA-Z0-9]", "", re.sub(r"\s+", "-", row[1]))])
        self.songs = self.songs[:self.lim]


    def delete_track(self, track_id):
        path = f"{self.folder_path}{track_id}.mp3"
        if exists(path):
            remove(path)
        
    def get_ydl_opts(self, path):
        return {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": f"{path}/%(id)s.%(ext)s",
            "ignoreerrors": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "320",
                }
            ],
        }
        
    def download_tracks(self):
        print(f"\033[1m\033[33m[info] Downloading {self.lim} tracks \033[0m")
        with YoutubeDL(self.get_ydl_opts(self.folder_path)) as ydl:
            for i, track in enumerate(self.songs):
                
                print_progress_bar(-1, i, len(self.songs))
                query = "-".join(track)
                title=f"{track[0][:7].lower()}_{track[1][:7].lower()}"
                
                html = rq.urlopen(f"https://www.youtube.com/results?search_query={query}")
                video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())

                if video_ids:
                    url = "https://www.youtube.com/watch?v=" + video_ids[0]
                    metadata = ydl.extract_info(url, download=False)
                    downloaded_track = ydl.download([url])
                    generate_spectrogram(self.folder_path, metadata["id"], title)
                    self.delete_track(metadata["id"])
    
    

if __name__ == "__main__":
    
    answer = input("Download songs[1] or images[2]?: ")
    if answer == "1":
        audio_downloader()

    else:   
        answer = input("Would you like to download the whole dataset [1] or download a single image [2]? ")
        if answer == "1":
            download_entire_file()

        else:
            print("Selecting the second option...")
            download_single_image("https://i.scdn.co/image/ab67616d0000b273eb02d5af28b4cbed922cb2ea")
        