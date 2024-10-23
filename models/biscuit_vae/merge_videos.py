from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
clips = []

for i in os.listdir('/Data/dibyanayan/CRL/BISCUIT/outputs/fragments/'):
    v_path = os.path.join('/Data/dibyanayan/CRL/BISCUIT/outputs/fragments/', i)
    clips.append(VideoFileClip(v_path))


final_clip = concatenate_videoclips(clips)
final_clip.write_videofile(os.path.join('/Data/dibyanayan/CRL/BISCUIT/outputs/fragments/', "final.mp4"))

