import torch
import torchaudio
from back import EmotionRecognitionModel  
import random
import os

def load_model(model_path):
    model = EmotionRecognitionModel() 
    model.load_state_dict(torch.load(model_path)) 
    model.eval() 
    return model

def predict_mood(image_path, model):
    mood = random.choice(['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'])
    return mood

def suggest_song(mood):
    songs = {
        'Anger': ['intense_rock.mp3', 'heavy_metal.mp3'],
        'Contempt': ['cold_synthwave.mp3', 'dark_ambient.mp3'],
        'Disgust': ['unpleasant_song.mp3', 'disturbing_beat.mp3'],
        'Fear': ['tense_soundtrack.mp3', 'spooky_theme.mp3'],
        'Happiness': ['happy_song.mp3', 'joyful_tune.mp3'],
        'Neutral': ['ambient_music.mp3', 'calm_piano.mp3'],
        'Sadness': ['sad_song.mp3', 'melancholy_track.mp3'],
        'Surprise': ['upbeat_song.mp3', 'exciting_track.mp3']
    }
    
    song_choice = random.choice(songs.get(mood, ['default_song.mp3']))
    return song_choice

def play_song(song_path):
    waveform, sample_rate = torchaudio.load(song_path)
    torchaudio.save('output_song.wav', waveform, sample_rate)

if __name__ == "__main__":
    # Load the model
    model = load_model("emotion_recognition_model.pth")

    image_path = "input_image.png" 
    mood = predict_mood(image_path, model)
    
    suggested_song = suggest_song(mood)
    print(f"Suggested Song: {suggested_song}")
    
    play_song(suggested_song)
