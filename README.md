# SynaethesAI: Investigating Cross-Modal Generation of Album Art from Audio

This code contains multiple files which were used to investigate the capacity for generating album covers directly from audio. All python files can be found in the `src` folder.

This project aims to use various different models to convert audio inputs into image output. This is done majorily through the use of variational autoencoders, and the audio is converted to Mel-spectrogram format to generate the outputs.

**Disclaimer**: This code is not production-level software and does not include exception handling or robust error checking. It is provided solely to illustrate the training methods used for models. While the core logic remains consistent across systems, it may not run correctly on all machines without modification. It is not submitted with the intention of claiming additional software marks, rather to give an overall impression of the methods used for training.

# Contents
1. Data
    1. Dataset
    2. Image Downloads
    3. Audio Downloads
    4. Outputs
2. Code
3. Models

# Data
## Dataset
The full dataset is contained within Music.csv, found from [Kaggle](https://www.kaggle.com/datasets/jashanjeetsinghhh/songs-dataset) and created by Jashanjeet Singh. This is a `.csv` file containing 65,000+ songs with their related Spotify IDs and image links. This is the file that  `download.py` uses to download content. It must be called `Music.csv` to be recognised by the code.

## Image Downloads
All image downloads go into the `downloaded_images` folder. A collection of the images used for training is also included as a `.zip` file inside the `data` folder. When you unzip this folder, be sure to name it `downloaded_images` for the code to properly recognise it!

## Audio Downloads
Audio is downloaded using a YouTube API. They will be stored as `.npy` files inside of the spectrograms folder. When downloading, the audio will first be installed a the full song `.mp3` inside of `temp_song_archive`. The full audio downloads will delete themselves once the spectrogram has been generated. For the audio spectrograms to be used during training, there must be a corresponding image with the exact same name in `.jpeg` format. The only `Dataloader` for audio combines both audio and image, and to do this it uses filenames from both. For all audio-related sections, if there is an error it is likely because no corresponding image exists in `downloaded_images`. If you don't have an image, you can rename an existing image to the right name. 

## Outputs
All image outputs will be stored under the `outputs` folder. 

## Test Images
When generating reconstruction of images, ensure that the image you want to test a reconstruction of is inside of the `test_images` folder. All python files will scan this entire folder for images to scan, so make sure that you only have the images you want to generate with in here. 

# Code
All code can be found inside of `src`. This code was designed for my testing of the models, and so some sections such as running models and testing outputs may seem unclear at first glance. This README file aims to clear up any uncertainty in running programs. Before running any programs, create a `.env` file in the root of the project (same depth as this README and `requirements.txt`) containing the path to the root of this program. In my local project, this looks like:
```python3
root="C:/Users/chwah/Dropbox/Family/Christopher/University/Y3/Year Long Project/SynaesthesAI"
```
This root variable is used to navigate the file system used by the project. In `linux`, you can set the root to `../`.

If you wish to try and test any program with erroneous data, you will likely get errors (aside from a few caught cases). This code was not designed to be fully robust; if you give it bad information, you will get bad information back. Instead, please try and get inputs correct for the models you wish to train and test. Any queries for running the program can be sent to [cw2503@bath.ac.uk](mailto:cw2503@bath.ac.uk).

## `download.py`
This is the program used to download both audio and image. You should ensure that you have all of the required folders listed in the **data** section created. You will be initially asked if you wish to download songs or images. Both options will ask you how many entries you wish to download - the code will check to ensure that there are no duplicate downloads. Saved entries are named using this naming convention: `title[:7]_artist[:7]`, which has yet to generate any clashes. You will be able to see the progress of downloads in the terminal as they download.

## `custom_distributed`
Created by GitHub user `blank`, this is a custom module used for VQVAE2. You can find their original code [here](github.com)

## `networks.py`
This file contains all classes pertaining to machine learning models. Whilst it doesn't run anything itself, all of the code for LSTM, VAE, VQVAE, VQVAE2, etc is stored here. There are also additional classes here which were used for testing but not in the final report, such as `LatentGPT`. All code is cited within the report.

## `test.py`
This folder contains all the code for testing image reconstructions and generations from either the `audio-encoder` or the traditional `autoencoders`. When you run, there is the option to test either the audio-to-image conversion or standard image reconstruction. The models for audio-to-image are hard-coded into the program, and you can change this if the audio model you wish to use changes. There is the option for image reconstruction, where you must specify the model you wish to reconstruct with. If you select `all`, it automatically begins generating for all models found - this includes non-autoencoders. Once it begins generating with non-autoencoders, you should stop the program or either let it crash. There are also functions here which can be used to calculate MSE error between two images. 

## `train_audioencoder.py`
This program trains a VQVAE to convert audio into images. When run, the program will ask you to specify an image model to train the audio-encoder off of. When prompted for the audio model, come up with a name that reflects the original image model being trained off. The image size should equal that which the autoencoder was trained at for best results. You have the option to load a pretrained model if it already exists. Any models not trained as a vanilla VAE will be trained as VQVAE2/VQGAN-FT, so select this option for any non-VAE image models.

## `train_gpt.py`
This code was originally designed to test a third type of image sampling using transformers. However, any time the code is run I faced an `OutOfMemoryError`. It is unclear if this is due to bad memory management or if it is due to low-memory GPUs. Either way, **this code is not submitted with the intention of being marked**, hence why it is not included in the final report. Instead, it is there in case anyone is interested in how I tried to use transformers to sample the latent space. The main method involved training a GPT on the full dataset of all images for X epochs, and then refining it to audio input after being fully trained. The code should run given infinite memory, however must be subject to additional testing to ensure that audio codes can be used as conditioning for a transformer.

## `train_lstm.py`
This file is used for all LSTM-related sampling of the autoencoder, including training **and** testing. When you run this file, you have the option to load an image model, and your top and bottom latents. For best practice, you should name the latents `t_latents.pt` and `b_latents.pt` - this was only included for more freedom when testing. Extracting latents will use an autoencoder to generate latent codes for the full dataset of images. Training requires that you already have latents extracted. Decoding will generate a novel image using spectrograms found in the spectrograms folder. There is also the option to train an LSTM without using audio. This will use the full dataset of images found in `downloaded_images` to train. All LSTM files should be stored under `models/LSTM/~model_name` for good training. Some latent codes are hard-coded into `sample_latents`; you can change this to allow for your created latent codes.

## `train_vqgan.py`
This code trains a VQGAN on the full dataset of images. You have the option to use a VQVAE2 as a base for the VQGAN, which can be chosen by selecting `existing` model - or selecting the use of a `premade VQVAE` in the second stage. Both do the same thing; originally you could load a partially trained VQGAN but this functionality was removed for testing. The rest of the model will train directly after. 

## `train_vqvae.py`
This file trains VAE, VQVAE, and VQVAE2 models on images, as well as VQVAE models for audio reconstruction (originally intended for the transformer generation). The sequence of inputs is fairly self-explanatory after this - hierarchical means VQVAE2. The code will tell you once it has finished training.

## `utils.py`
One of the more abstract files here, this code contains all dataset code as well as the progress bar file and latent code extraction: basically all functions that don't fit into one of the other files. `CustomAudioImagePairing` will automatically pair images and audio by filename, whilst `CustomImageFolder` only extracts images. There are a handful of almost identical functions called `extract_latent_codes` or some variation thereupon. These all extract the codes needed to train the LSTMs. Ensure that you have the right folder system before attempting to download any latents.

# Models
All models are stored under the model folder. They are sorted into four categories: `128x128`, `256x256`, `AUDIO-ENCODING`, and `LSTM`. A naming convention is used to make identifying each of these models easier:

`VAE`: Traditional VAE model

`VQVAE`: Non-hierarchical VQVAE

`VQVAE2`: Hierarchical VQVAE

`VQGAN`: VQGAN trained without a pretrained VQVAE

`VQGAN-FT`: VQGAN trained using a pretrained VQVAE. 

The number at the end of the model names tells you what resolution the images used for training were. For example, `AUDIO-VQGAN-FT256` is a model that takes audio as an input, and then converts it to image and had been trained using a VQGAN which has been trained on 256x256 resolution images and a pretrained VQVAE2. For the `LSTM` folder, subfolders should be created for each unique LSTM tested. Example folders are already in the source code.  