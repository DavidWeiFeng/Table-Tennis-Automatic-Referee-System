import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import deque
from dataclasses import dataclass
from typing import Optional
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import os
import pyttsx3
import threading

@dataclass
class BallPosition:
    x: int
    y: int

class TextOverlay:
    def __init__(self):
        self.messages = {}  # 存储所有需要显示的消息
        
    def add_message(self, text, duration_frames, position=(100, 350), 
                   font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, 
                   color=(0, 0, 255), thickness=2):
        """
        添加一条需要显示的消息
        text: 显示的文字
        duration_frames: 持续的帧数
        position: 显示位置 (x,y)
        font: 字体
        font_scale: 字体大小
        color: 颜色 (B,G,R)
        thickness: 字体粗细
        """
        self.messages[text] = {
            'duration': duration_frames,
            'position': position,
            'font': font,
            'font_scale': font_scale,
            'color': color,
            'thickness': thickness
        }
    
    def draw(self, frame):
        """
        在帧上绘制所有活跃的消息
        """
        # 使用列表存储需要删除的消息
        to_remove = []
        
        for text, props in self.messages.items():
            if props['duration'] > 0:
                cv2.putText(frame, text, 
                           props['position'],
                           props['font'],
                           props['font_scale'],
                           props['color'],
                           props['thickness'])
                props['duration'] -= 1
            else:
                to_remove.append(text)
        
        # 删除已经显示完的消息
        for text in to_remove:
            self.messages.pop(text)
            
        return frame


class TableTennisDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化语音引擎
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # 设置语速
        self.engine.setProperty('volume', 1.0)  # 设置音量
        
        # 初始化文本显示对象
        self.text_overlay = TextOverlay()
        
        self.pre_centerx = 0
        self.pre_centery = 0
        self.first_ball_detected = False
        self.points = []  # 存储选取的四个点，上左，上右，下左，下右
        self.first_frame = None  # 存储第一帧图像
        self.model = YOLO('best11n.engine')
        self.camera_fps = 60 #相机帧率
        self.positions = deque(maxlen=10)  # 用于存储最近15帧的位置   
        self.serve_timer_active = False  # 发球计时器是否激活
        self.frame_count = 0  # 帧计数器
        self.serve_detected = False  # 是否检测到发球
        self.serve_status = None  # 发球选手(left/right)
        self.expected_position = None  # 期望的下一个落点
        self.left_score = 0
        self.right_score = 0
        self.current_game = 0  # 当前是第几局，从0开始
        # ... (保持其他初始化变量不变)
        
        # 添加新的变量
        self.is_running = False
        self.score_history = []  # 用于记录得分历史
        self.video_path = ""
        self.model_path = 'best11n.engine'
        
        self.setup_ui()
    def reset_serve_state(self):
        """重置发球相关的状态"""
        self.serve_timer_active = False
        self.serve_detected = False
        self.serve_status = None
        self.expected_position = None
        self.frame_count = 0
        
    def add_position(self, x: int, y: int):
        """检测发球"""
        current_pos = BallPosition(x=x, y=y)
        self.positions.append(current_pos)        
        
        if len(self.positions) == self.positions.maxlen:
            # 检查是否所有帧都在下降
            is_descending = all(
                self.positions[i-1].y > self.positions[i].y 
                for i in range(1, len(self.positions))
            )
            # 检查是否在边界外
            
            if is_descending:
                big_x=max(current_pos.x,self.positions[0].x)
                small_x=min(current_pos.x,self.positions[0].x)
                # 计算偏移量分比
                offset_percent=(big_x-small_x)/big_x
                # 如果偏移量百分比小于0.1，则认为是发球
                if offset_percent<0.1:
                     # 确定发球方
                    if current_pos.x < self.mid_boundary:
                        self.serve_status = "left"
                        self.expected_position = "left"  # 左侧发球首先要打到左侧
                    elif current_pos.x > self.mid_boundary:
                        self.serve_status = "right"
                        self.expected_position = "right"  # 右侧发球首先要打到右侧
                    return True
        return False

    def checkBallInOut(self,center_x,center_y):
        if center_y-self.pre_centery<0 and self.top_boundary-20<self.pre_centery<self.bottom_boundary+20:
            if self.left_boundary<center_x<self.mid_boundary:
                return "left"
            elif self.mid_boundary<center_x<self.right_boundary:
                return "right"

        else :
            return 

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                # 在图像上画点
                cv2.circle(self.first_frame, (x, y), 5, (0, 0, 255), -1)
                # 显示坐标
                cv2.putText(self.first_frame, f'({x},{y})', (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('First Frame', self.first_frame)
                
                if len(self.points) == 4:
                    print("四个点的坐标：", self.points)
                    # 计算四个点的坐标
                    self.top_left = self.points[0]
                    self.top_right = self.points[1]
                    self.bottom_left = self.points[2]
                    self.bottom_right = self.points[3]
                    #左边界
                    self.left_boundary = self.bottom_left[0]
                    #右边界
                    self.right_boundary = self.bottom_right[0]
                    #上边界
                    self.top_boundary = self.top_left[1]
                    #下边界
                    self.bottom_boundary = self.bottom_left[1]
                    #中边界
                    self.mid_boundary = int((self.left_boundary+self.right_boundary)/2)
                    self.table_width = self.right_boundary-self.left_boundary
                    # 等待按键后继续
                    cv2.waitKey(0)
                    cv2.destroyWindow('First Frame')

    def process_frame(self, frame):

        # 如果发球已检测到，开始计数
        if self.serve_detected:
            self.frame_count += 1
            remaining_frames = self.camera_fps*2 - self.frame_count
            cv2.putText(frame, f"Remaining: {remaining_frames}", (100,300), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # 超时检查
            if self.frame_count >= self.camera_fps*2:
                    # 判定得分
                    winner = "right" if self.expected_position == "right" else "left"
                    ## 更新得分
                    if winner=="right":
                        self.right_score += 1
                        self.score_history.append("right")
                    else:
                        self.left_score += 1
                        self.score_history.append("left")
                    # cv2.putText(frame, f"{winner} player scores!", (100,350), 
                    #           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    self.text_overlay.add_message(f"{winner} player scores!", duration_frames=30)

                    # 重置所有状态
                    self.reset_serve_state()
                    # 更新UI显示和播报比分
                    self.update_score_display(speak=True)
        # 显示得分
        cv2.putText(frame, f"{self.left_score}:{self.right_score}", (100,400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # 只更新UI显示，不播报
        self.update_score_display(speak=False)
        
        results = self.model.predict(frame, stream=True)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = int((x1+x2)/2)
                    center_y = int((y1+y2)/2)
                    # 显示球心坐标
                    center = f'{center_x},{center_y}'
                    
                    if not self.first_ball_detected:
                        self.first_ball_detected = True
                        self.pre_centerx = center_x
                        self.pre_centery = center_y
                    # cv2.putText(frame, f"centery:{center_y}", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    # cv2.putText(frame, f"pre_centery:{self.pre_centery}", (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    
                    # 检测是否发球
                    if not self.serve_detected:
                        if self.add_position(center_x, center_y):
                            self.serve_detected = True #检测到发球后，就不再检测发球了，直到得分后才重新检测发球
                            self.serve_timer_active = True #激活帧计时器
                            self.frame_count = 0
                            # cv2.putText(frame, f"Serve: {self.serve_status}", (100,150), 
                            #           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            self.text_overlay.add_message(f"Serve: {self.serve_status}", duration_frames=30)
                    
                    # 发球后的追踪逻辑
                    if self.serve_timer_active:
                        # 检测球是否上台
                        position = self.checkBallInOut(center_x, center_y)
                        if position:
                            cv2.putText(frame, f"Position: {position}", (100,100), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            
                            if position == self.expected_position:
                                # 更新下一个期望的落点
                                self.expected_position = "right" if position == "left" else "left"
                                # 重置计时器
                                self.frame_count = 0
                        

                    # 更新球的位置
                    self.pre_centery = center_y
                    self.pre_centerx = center_x
                    
                    label = f'ball: {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    cv2.putText(frame, center, (center_x, center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    frame = self.text_overlay.draw(frame)

        return frame

    def infer_video(self, video_path,output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法读取视频")
            return
          # 获取视频的宽、高和帧率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        ret, self.first_frame = cap.read()
        if not ret:
            print("无法读取第一帧")
            return
        
        cv2.namedWindow('First Frame')
        cv2.setMouseCallback('First Frame', lambda event, x, y, flags, param: self.mouse_callback(event, x, y, flags, param))
        #初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4格式
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cv2.imshow('First Frame', self.first_frame)
        while len(self.points) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame)
            cv2.imshow("yolo", frame)
            # 写入处理后的帧到视频
            video_writer.write(frame)
            
            # key = cv2.waitKey(0) & 0xFF
            # if key == ord('q'):
            #     break
            # elif key == ord('k'):
            #     continue
                
        cap.release()
        video_writer.release()  # 释放视频写入器
        cv2.destroyAllWindows()
        
    def setup_ui(self):
        """设置GUI界面"""
        self.setWindowTitle('乒乓球自动裁判系统')
        self.setGeometry(100, 100, 1600, 1000)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()  # 主布局为水平布局
        
        # 左侧控制面板
        left_panel = QVBoxLayout()
        
        # 控制按钮
        self.video_btn = QPushButton('选择视频')
        self.model_btn = QPushButton('选择模型')
        self.start_btn = QPushButton('开始比赛')
        self.stop_btn = QPushButton('结束比赛')
        
        self.video_btn.clicked.connect(self.select_video)
        self.model_btn.clicked.connect(self.select_model)
        self.start_btn.clicked.connect(self.start_game)
        self.stop_btn.clicked.connect(self.stop_game)
        
        # 添加按钮到左侧面板
        left_panel.addWidget(self.video_btn)
        left_panel.addWidget(self.model_btn)
        left_panel.addWidget(self.start_btn)
        left_panel.addWidget(self.stop_btn)
        left_panel.addStretch()
        
        # 右侧主要内容区域
        right_content = QVBoxLayout()
        
        # 顶部玩家和得分区域
        top_section = QHBoxLayout()
        
        # Player 1 区域
        player1_section = QVBoxLayout()
        player1_label = QLabel('PLAYER 1')
        player1_label.setAlignment(Qt.AlignCenter)
        self.left_score_btn = QPushButton('得分')
        self.left_score_btn.clicked.connect(lambda: self.add_score('left'))
        player1_section.addWidget(player1_label)
        player1_section.addWidget(self.left_score_btn)
        
        # 中间得分显示
        score_display = QHBoxLayout()
        self.score_label = QLabel('00 : 00')
        self.score_label.setStyleSheet('font-size: 48px; font-weight: bold;')
        self.score_label.setAlignment(Qt.AlignCenter)
        score_display.addWidget(self.score_label)
        
        # 中间得分区域（直布局，包含得分显示和撤销按钮）
        score_section = QVBoxLayout()
        score_section.addLayout(score_display)
        
        # 添加撤销按钮
        self.undo_btn = QPushButton('此分无效')
        self.undo_btn.clicked.connect(self.undo_score)
        self.undo_btn.setFixedWidth(100)  # 设置按钮宽度
        undo_btn_layout = QHBoxLayout()  # 用于居中按钮
        undo_btn_layout.addStretch()
        undo_btn_layout.addWidget(self.undo_btn)
        undo_btn_layout.addStretch()
        score_section.addLayout(undo_btn_layout)
        
        # Player 2 区域
        player2_section = QVBoxLayout()
        player2_label = QLabel('PLAYER 2')
        player2_label.setAlignment(Qt.AlignCenter)
        self.right_score_btn = QPushButton('得分')
        self.right_score_btn.clicked.connect(lambda: self.add_score('right'))
        player2_section.addWidget(player2_label)
        player2_section.addWidget(self.right_score_btn)
        
        # 添加到顶部区域
        top_section.addLayout(player1_section)
        top_section.addLayout(score_section)  # 改为添加score_section而不是score_display
        top_section.addLayout(player2_section)
        
        # 比赛记录表格
        self.score_table = QTableWidget()
        self.score_table.setRowCount(2)
        self.score_table.setColumnCount(7)
        self.score_table.setHorizontalHeaderLabels(['1', '2', '3', '4', '5', '6', '7'])
        self.score_table.setVerticalHeaderLabels(['PLAYER1', 'PLAYER2'])
        
        # 设置表格的每个单元格为0
        for i in range(2):
            for j in range(7):
                item = QTableWidgetItem('0')
                item.setTextAlignment(Qt.AlignCenter)
                self.score_table.setItem(i, j, item)
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setMinimumSize(1280, 720)
        self.video_label.setStyleSheet('background-color: black;')
        self.video_label.setAlignment(Qt.AlignCenter)
        
        # 添加所有组件到右侧内容区域
        right_content.addLayout(top_section)
        right_content.addWidget(self.score_table)
        right_content.addWidget(self.video_label)
        
        # 将左侧面板和右侧内容添加到主布局
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_content, stretch=4)  # 右侧内容区域占据更多
        
        main_widget.setLayout(main_layout)
        
    def update_score_display(self, speak=True):
        """更新得分显示"""
        self.score_label.setText(f'{self.left_score:02d} : {self.right_score:02d}')
        # 只在需要时播报比分
        if speak:
            self.speak_score()
        
    def add_score(self, player):
        """添加得分"""
        if player == 'left':
            self.left_score += 1
        else:
            self.right_score += 1
        self.score_history.append(player)
        self.update_score_display(speak=True)
        
    def undo_score(self):
        """撤销上一次得分"""
        if self.score_history:
            last_scorer = self.score_history.pop()
            if last_scorer == 'left':
                self.left_score -= 1
            else:
                self.right_score -= 1
            self.update_score_display(speak=True)
        
    def select_model(self):
        """选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Engine Files (*.engine)")
        if file_path:
            self.model_path = file_path
            self.model = YOLO(self.model_path)
            # 更新按钮文本为文件名
            file_name = os.path.basename(file_path)
            self.model_btn.setText(file_name)
            
    def select_video(self):
        """选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.video_path = file_path
            # 更新按钮文本为文件名
            file_name = os.path.basename(file_path)
            self.video_btn.setText(file_name)
            
    def start_game(self):
        """开始比赛"""
        if not self.video_path:
            QMessageBox.warning(self, "警告", "请先选择视频文件！")
            return
        if self.current_game >= 7:
            QMessageBox.warning(self, "警告", "比赛已经结束！")
            return
        self.is_running = True
        self.process_video()
        
    def stop_game(self):
        """结束比赛"""
        if self.is_running and self.current_game < 7:  # 确保在比赛进行中且未超过7局
            # 更新比赛记录表格
            left_item = QTableWidgetItem(str(self.left_score))
            right_item = QTableWidgetItem(str(self.right_score))
            left_item.setTextAlignment(Qt.AlignCenter)
            right_item.setTextAlignment(Qt.AlignCenter)
            self.score_table.setItem(0, self.current_game, left_item)  # PLAYER1的得分
            self.score_table.setItem(1, self.current_game, right_item)  # PLAYER2的得分
            self.current_game += 1  # 更新局数
            
            # 重置当前局比分
            self.left_score = 0
            self.right_score = 0
            self.score_history.clear()  # 清空得分历史
            self.update_score_display(speak=True)  # 更新显示并播报
            
        self.is_running = False
        
    def process_video(self):
        """处理视频"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频文件！")
            return
            
        ret, self.first_frame = cap.read()
        if not ret:
            QMessageBox.warning(self, "错误", "无法读取第一帧！")
            return
            
        # 显示第一帧并等待用户选择四个点
        cv2.namedWindow('First Frame')
        cv2.setMouseCallback('First Frame', self.mouse_callback)
        cv2.imshow('First Frame', self.first_frame)
        
        while len(self.points) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
                
        try:
            cv2.destroyWindow('First Frame')
        except:
            pass
        
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = self.process_frame(frame)
            
            # 将OpenCV图像转换为Qt图像并显示
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_image = qt_image.scaled(self.video_label.size(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(QPixmap.fromImage(scaled_image))
            
            QApplication.processEvents()
            
        cap.release()

    def speak_score(self):
        """语音播报当前比分"""
        text = f"{self.left_score}比{self.right_score}"
        # 在新线程中播放语音，避免阻塞主线程
        threading.Thread(target=self.engine.say, args=(text,), daemon=True).start()
        threading.Thread(target=self.engine.runAndWait, daemon=True).start()

def main():
    app = QApplication(sys.argv)
    detector = TableTennisDetector()
    detector.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()