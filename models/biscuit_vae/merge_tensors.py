import torch
import os
import cv2
import numpy as np


def find_box(video_dir):
    sequence_length = 1
    box_sizes = []
    video_files = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith(('.mp4', '.avi'))][48:50]
    
    for video_file in video_files:
        # Use cv2 to read video metadata and extract frame count
        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
        cap.release()  # Release the video capture object

        # Calculate the number of frame sequences for this video
        box_sizes.append(max(0, (frame_count - sequence_length + 1)))

    return box_sizes
    
box_sizes = find_box('/Data/dibyanayan/CRL/BISCUIT/data/tvsumm/ydata-tvsum50-v1_1/video/')

print(box_sizes)


clips = []

for i in range(len(os.listdir('/Data/dibyanayan/CRL/BISCUIT/outputs/fragments_nf_latent/'))):
    t_path = os.path.join('/Data/dibyanayan/CRL/BISCUIT/outputs/fragments_nf_latent/causal_latent_{}.pt'.format(i))
    clips.append(torch.load(t_path))

clips = torch.cat(clips, dim=0)

print(clips.shape)

first_clip, second_clip = clips[:box_sizes[0]-1, :],  clips[box_sizes[0]-1:, :]

print(first_clip.shape, second_clip.shape)



frames = torch.load('/Data/dibyanayan/CRL/BISCUIT/experiments/all_frames_tvsumm.pt')

first_frame, second_frame = frames[:box_sizes[0], :],  frames[box_sizes[0]:, :]

print(frames.shape, first_frame.shape, second_frame.shape)

# print(0/0)





def point_line_distance(point, start, end):
    """
    Calculate the perpendicular distance of a point from a line defined by the start and end points in any dimension.
    Args:
        point: A tensor representing the point to be checked (40d).
        start: A tensor representing the starting point of the line (40d).
        end: A tensor representing the ending point of the line (40d).
    
    Returns:
        The perpendicular distance of the point to the line (scalar).
    """
    # Line vector: end - start
    line_vec = end - start
    
    # Vector from start to point
    point_vec = point - start
    
    # Project point_vec onto line_vec and calculate the perpendicular component
    line_length_squared = torch.dot(line_vec, line_vec)
    if line_length_squared == 0:
        return torch.norm(point - start)  # Line is a single point
    
    projection = torch.dot(point_vec, line_vec) / line_length_squared
    projection_point = start + projection * line_vec
    
    # Return the distance from the point to the projection on the line
    return torch.norm(point - projection_point)

def rdp(points, epsilon):
    """
    Simplify a curve using the Ramer-Douglas-Peucker algorithm.
    
    Args:
        points: A tensor of size (2000, 40) where each row represents a point in 40-dimensional space.
        epsilon: The distance threshold to determine which points to keep.
        
    Returns:
        A list of indices of the points that should be kept.
    """
    # Helper function for recursion
    def rdp_recursive(start_idx, end_idx):
        if start_idx >= end_idx:
            return []
        
        # Find the point with the maximum distance from the line between the start and end points
        start_point = points[start_idx]
        end_point = points[end_idx]
        max_distance = -1
        index_of_max = start_idx
        
        for i in range(start_idx + 1, end_idx):
            distance = point_line_distance(points[i], start_point, end_point)
            if distance > max_distance:
                max_distance = distance
                index_of_max = i
        
        # If the maximum distance is greater than epsilon, recursively simplify
        if max_distance > epsilon:
            # Recursively simplify the sections left and right of the farthest point
            left_indices = rdp_recursive(start_idx, index_of_max)
            right_indices = rdp_recursive(index_of_max, end_idx)
            
            # Combine results: left + max point + right
            return left_indices + [index_of_max] + right_indices
        else:
            # No point is far enough; keep only the endpoints
            return []
    
    # Call the recursive function
    indices = [0] + rdp_recursive(0, points.size(0) - 1) + [points.size(0) - 1]
    return indices


epsilon = 4.0  # Set an appropriate threshold for your use case
first_indices = rdp(first_clip, epsilon)
second_indices = rdp(second_clip, epsilon)

print(len(first_indices))
print(len(second_indices))


def get_video_summ_extractive(video_tensor, batch_idx):

    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()

    

    # Normalize the pixel values to [0, 255] (assuming the tensor is in range [-1, 1] or [0, 1])
    video_np = (255 * (video_np - video_np.min()) / (video_np.max() - video_np.min())).astype(np.uint8)

    

    # Video parameters
    height, width, channels = video_np[0].shape
    fps = 30  # Frames per second
    output_file = '/Data/dibyanayan/CRL/BISCUIT/outputs/summ_extractive/{}.mp4'.format(batch_idx)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in video_np:
        #print(frame)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

    print(f"Video saved as {output_file}")

get_video_summ_extractive(first_frame[torch.tensor(first_indices)+1, :], 'summ_49')
get_video_summ_extractive(second_frame[torch.tensor(second_indices), :], 'summ_50')


get_video_summ_extractive(first_frame, 'full_49')
get_video_summ_extractive(second_frame, 'full_50')