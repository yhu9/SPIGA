import os

from spiga.demo.app import video_app

# MP4 input path: Webcam recorded video or uploaded one.
# video_path = '/content/<path_to_your_video>'
# video_path = webcam_video_path
video_path = "/home/masahu/data/MEAD/extracted/M007/video/front/neutral/level_1/029.mp4"
output_path = 'content/output'  # Processed video storage

# Process video
video_app(video_path,
          # Choices=['wflw', '300wpublic', '300wprivate', 'merlrav']
          spiga_dataset='merlrav',
          # Choices=['RetinaSort', 'RetinaSort_Res50']
          tracker='RetinaSort',
          save=True,
          output_path=output_path,
          visualize=False,
          plot=['landmarks'])
