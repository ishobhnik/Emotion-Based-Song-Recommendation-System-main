// Main file that combines everything
import { getMoodFromFace } from './emotionDetection.js';
import { searchTracks } from './spotifyAPI.js';

async function suggestSongsBasedOnMood(faceImage) {
  const mood = await getMoodFromFace(faceImage);
  const emotionQueries = {

    happiness: 'happy upbeat',
    contempt:'sad',
    sad: 'calm relaxing',
    anger: 'energetic intense',
    fear: 'suspenseful dramatic',
    neutral: 'ambient',
    surprise: 'exciting',
    disgust:'rap'
  };

  const tracks = await searchTracks(emotionQueries[mood]);

  tracks.forEach(track => {
    console.log(`${track.name} by ${track.artists.map(artist => artist.name).join(', ')}`);
    console.log(`Listen here: ${track.external_urls.spotify}`);
  });
}
suggestSongsBasedOnMood('face_image_data_here');
