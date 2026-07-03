import cv2
import os
import glob
import time

# 设置图像文件所在的目录
image_dir = './outputs/20260628_162448'  # 替换为你的图像文件所在目录

# 获取所有图像文件并按名称排序
images = glob.glob(os.path.join(image_dir, 'intent_graph_window_*.png'))  # 根据实际格式修改扩展名
images.sort()  # 确保文件按正确顺序排序

# 检查是否有找到图像
if not images:
    print("未找到任何图像文件，请检查路径和文件名")
    exit()

# 读取第一帧以获取视频尺寸
first_frame = cv2.imread(images[0])
height, width, layers = first_frame.shape

# 设置裁剪后的宽度（保持高度不变）
crop_width = width - 100  # 假设裁剪掉右侧100像素，你可以根据实际情况调整

# 计算所需的帧率：每张图片显示0.5秒，意味着每秒2帧
fps = 2.0  # 每秒2帧，这样每张图片会显示0.5秒

# 设置视频编码器和输出文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = 'output_video.mp4'

# 创建视频写入对象，使用裁剪后的真实尺寸 (width, height)
out = cv2.VideoWriter(output_video, fourcc, fps, (crop_width, height))

# 将所有图像写入视频
for image in images:
    try:
        frame = cv2.imread(image)
        if frame is None:
            print(f"警告: 无法读取图像 {image}")
            continue

        # 裁剪图像（从左侧开始，裁剪到指定宽度）
        cropped_frame = frame[:, :crop_width]

        # 确保裁剪后的尺寸一致
        if cropped_frame.shape != (height, crop_width, layers):
            print(f"警告: 图像 {image} 裁剪后尺寸不一致，跳过")
            continue

        # 每张图片写入1次（因为fps=2，所以每张图片会显示0.5秒）
        out.write(cropped_frame)
    except Exception as e:
        print(f"处理图像 {image} 时发生错误: {str(e)}")

# 释放资源
out.release()
cv2.destroyAllWindows()

# 验证输出文件
if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
    print(f"视频已成功生成，文件名: {output_video}")
    # 获取视频信息
    cap = cv2.VideoCapture(output_video)
    if cap.isOpened():
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / actual_fps  # 计算视频总时长
        print(f"视频信息: 帧率={actual_fps:.2f}fps, 总帧数={frame_count}, 总时长={duration:.2f}秒")
        cap.release()
else:
    print("错误: 视频生成失败，输出文件无效")
