import cv2
import os
from glob import glob


def images_to_video(image_folder, output_video, fps=30):
    # 获取所有图片路径，并按文件名排序
    images = sorted(glob(os.path.join(image_folder, '*')))

    if not images:
        print("No images found in the folder.")
        return

    # 读取第一张图片以获取尺寸
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # 定义视频编解码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}, unable to read.")
            continue
        video_writer.write(img)

    video_writer.release()
    print(f"Video saved as {output_video}")


# 使用示例
image_folder = 'recording/episode_82'  # 替换为你的图片文件夹路径
output_video = 'cyclist.mp4'  # 输出视频文件名
fps = 30  # 帧率

images_to_video(image_folder, output_video, fps)
