import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import urllib.request
import regex as re

class cover_art_downloader:

    def __init__(self, playlist_link):
        self.playlist_link = playlist_link

    def set_playlist(self, playlist_link):
        self.playlist_link = playlist_link

    def begin_download(self, folder="Covers"):
        spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
            client_id="34ac0965ba9546f091b6687504317dd4",
            client_secret="62ed44dc10994f83a7092d27156a5feb"))
        results = spotify.playlist(self.playlist_link)

        print("Length: ", len(results["tracks"]["items"]))

        for item in results["tracks"]["items"]:

            name = re.sub(r"[?!']", '', item["track"]["name"])
            imgURL = item["track"]["album"]["images"][0]["url"]
            urllib.request.urlretrieve(imgURL, rf"{folder}/{name}.jpeg")