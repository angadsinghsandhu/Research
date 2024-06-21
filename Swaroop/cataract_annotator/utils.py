import subprocess
import cv2
from cataract_annotator.audio_processing import save_audio
from cataract_annotator.video_processing import frames, fps, frame_size

def save_annotation_files(user_name, user_level):
    output_video_filename = f"{user_name}_{user_level}_output_video"
    output_audio_filename = f"{user_name}_{user_level}_output_audio"

    output_video_path = f"{output_video_filename}.mp4"
    output_audio_path = f"{output_audio_filename}.wav"

    # Save video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    for frame in frames:
        out.write(frame)
    out.release()

    # Save audio
    save_audio(output_audio_path)

    # Combine audio and video
    combine_audio_video(output_video_path, output_audio_path)

def combine_audio_video(output_video_path, output_audio_path, audio_delay=-1):
    command = [
        'ffmpeg',
        '-i', output_video_path,
        '-itsoffset', str(audio_delay),
        '-i', output_audio_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-shortest',
        output_video_path.replace('.mp4', '_annotated.mp4')
    ]
    subprocess.run(command)
