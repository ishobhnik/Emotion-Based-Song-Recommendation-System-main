import torch
import torchaudio
from PIL import Image
import torchvision.transforms as transforms
import random
import os

class EmotionRecognitionModel(torch.nn.Module):
    def __init__(self):
        super(EmotionRecognitionModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
        self.fc1 = torch.nn.Linear(64 * 224 * 224, 8) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x

def load_model(model_path):
    model = EmotionRecognitionModel()  
    model.load_state_dict(torch.load(model_path)) 
    model.eval() 
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def predict_mood(image_path, model):
    image = preprocess_image(image_path)
    image_tensor = image.unsqueeze(0) 
    with torch.no_grad():
        output = model(image_tensor)
    
    _, predicted_class = torch.max(output, 1)
    emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    mood = emotions[predicted_class.item()]
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
    model = load_model("emotion_recognition_model.pth")
    image_path = "tryimage4.png" 
    mood = predict_mood(image_path, model)
    suggested_song = suggest_song(mood)
    print(f"Suggested Song: {suggested_song}")
    play_song(f'songs/{suggested_song}')
