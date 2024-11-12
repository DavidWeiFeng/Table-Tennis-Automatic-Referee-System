from ultralytics import YOLO
import cv2
import time
import numpy as np

def process_video(video_path, output_path=None):
    # 加载YOLOv8姿态检测模型
    
    model = YOLO("yolo11n-pose.pt")  # load an official model

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
        result = results[0]
        
        # 获取原始图像的副本
        annotated_frame = frame.copy()
        
        if len(result.boxes) > 0:
            # 计算每个边界框的面积
            boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # 计算面积
            
            # 找到两个最大边界框的索引
            num_boxes = min(2, len(areas))  # 确保不超过检测到的人数
            max_indices = areas.argsort()[-num_boxes:][::-1]  # 获取最大的两个索引
            
            # 为两个人设置不同的颜色
            colors = [(0, 0, 255), (255, 0, 0)]  # 红色和蓝色

            if num_boxes == 2:
                # 获取两个边界框的x坐标（使用边界框的中心点x坐标）
                box1_center_x = (boxes[max_indices[0]][0] + boxes[max_indices[0]][2]) / 2
                box2_center_x = (boxes[max_indices[1]][0] + boxes[max_indices[1]][2]) / 2
                
                # 根据x坐标排序，确保左边的框对应索引0，右边的框对应索引1
                if box1_center_x > box2_center_x:
                    max_indices = max_indices[::-1]  # 交换两个索引的顺序
            
            # 绘制骨架连接定义
            skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                       [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
            
            # 为每个选中的人绘制姿态
            for idx, box_idx in enumerate(max_indices):
                
                color = colors[idx]  # 现在左边的人一定是红色(idx=0)，右边的人一定是蓝色(idx=1)
                
                # 获取边界框中心点x坐标（用于调试）
                box_center_x = (boxes[box_idx][0] + boxes[box_idx][2]) / 2
                
                # 可以添加标签显示位置（可选）
                position_label = "Left" if idx == 0 else "Right"                
                
                # 绘制边界框
                box = boxes[box_idx]
                cv2.rectangle(annotated_frame, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            color, 2)
                
                # 绘制关键点和骨架
                if result.keypoints is not None:
                    keypoints = result.keypoints[box_idx].data.cpu().numpy()[0]
                    
                    # 绘制关键点
                    for kp in keypoints:
                        x, y, conf = kp
                        if conf > 0.5:  # 只绘制置信度高的关键点
                            cv2.circle(annotated_frame, (int(x), int(y)), 5, color, -1)
                    
                    # 绘制骨架连接
                    for connection in skeleton:
                        kp1, kp2 = connection
                        if keypoints[kp1-1][2] > 0.5 and keypoints[kp2-1][2] > 0.5:
                            pt1 = (int(keypoints[kp1-1][0]), int(keypoints[kp1-1][1]))
                            pt2 = (int(keypoints[kp2-1][0]), int(keypoints[kp2-1][1]))
                            cv2.line(annotated_frame, pt1, pt2, color, 2)
                
                # 添加标签（可选）
                cv2.putText(annotated_frame, f'Person {idx+1}', 
                          (int(box[0]), int(box[1]-10)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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