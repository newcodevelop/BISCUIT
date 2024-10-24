


from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

# video_dir = '/Data/dibyanayan/CRL/BISCUIT/data/tvsumm/ydata-tvsum50-v1_1/video'

# video_files = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith(('.mp4', '.avi'))]

# print(video_files[48:])

# clips = []
# for i in video_files:
#     clips.append(VideoFileClip(i))

# final_clip = concatenate_videoclips(clips, method='compose')


# final_clip.write_videofile(os.path.join('/Data/dibyanayan/CRL/BISCUIT/outputs/fragments/', "final_op.mp4"), fps=24, remove_temp=False)



# print(0/0)


clips = []

for i in range(len(os.listdir('/Data/dibyanayan/CRL/BISCUIT/outputs/fragments/')))[236:400]:
    v_path = os.path.join('/Data/dibyanayan/CRL/BISCUIT/outputs/fragments/output_video_{}.mp4'.format(i))
    
    try:
        v_file = VideoFileClip(v_path)
        clips.append(v_file)
    except Exception as e:
        print('{} th frame not processed due to {}'.format(i,e))
        continue
    


final_clip = concatenate_videoclips(clips, method='compose')


final_clip.write_videofile(os.path.join('/Data/dibyanayan/CRL/BISCUIT/outputs/fragments/', "final.mp4"), fps=24, remove_temp=False)

