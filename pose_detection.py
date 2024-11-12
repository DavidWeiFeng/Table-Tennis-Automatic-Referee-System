from ultralytics import YOLO
import cv2
import time
import numpy as np

def process_video(video_path, output_path=None):
    # 加载YOLOv8姿态检测模型
    model = YOLO('yolov8n-pose.pt')  # 或者使用 yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 设置输出视频
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 处理视频帧
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 运行YOLOv8推理
        results = model.predict(frame, conf=0.5)
        
        # 在帧上绘制结果
        annotated_frame = results[0].plot()

        # 显示处理后的帧
        cv2.imshow("YOLOv8 姿态检测", annotated_frame)

        # 保存处理后的帧
        if output_path:
            out.write(annotated_frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设置视频路径
    video_path = r'videos\WeChat_20241112154837.mp4'  # 替换为你的视频文件路径
    output_path = "output.mp4"  # 输出视频路径（可选）
    
    process_video(video_path, output_path) 