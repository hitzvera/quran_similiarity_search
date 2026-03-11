from pydub import AudioSegment
import os

def convert_mp3_to_wav(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.mp3'):
            path = os.path.join(source_dir, filename)
            audio = AudioSegment.from_mp3(path)

            audio = audio.set_frame_rate(16000).set_channels(1)

            target_path = os.path.join(target_dir, filename.replace('.mp3', '.wav'))
            audio.export(target_path, format='wav')
    
    