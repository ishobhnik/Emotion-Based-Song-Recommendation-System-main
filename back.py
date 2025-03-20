import torch
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import io

class EmotionCNN(torch.nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        
        self.fc1 = torch.nn.Linear(128 * 6 * 6, 128)
        self.fc2 = torch.nn.Linear(128, 8)
        
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

model = EmotionCNN()

checkpoint_path = "emotion_recognition_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            _, predicted_class = torch.max(output, 1)
            emotion = predicted_class.item()

        emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        return jsonify({'emotion': emotions[emotion]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
