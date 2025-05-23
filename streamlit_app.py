import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
import os
import tempfile
from pathlib import Path
import time

def process_video(video_path, model, confidence, iou, frame_interval):
    """处理视频文件并返回处理后的视频路径和预览帧"""
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算实际要处理的帧数
    processed_fps = fps // frame_interval
    actual_total_frames = total_frames // frame_interval
    
    # 创建临时文件保存处理后的视频
    temp_dir = tempfile.mkdtemp()
    output_path = str(Path(temp_dir) / "output.mp4")
    preview_path = str(Path(temp_dir) / "preview.mp4")
    
    # 设置视频写入器，使用处理后的帧率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, processed_fps, (width, height))
    preview_out = cv2.VideoWriter(preview_path, fourcc, processed_fps, (width, height))
    
    # 创建进度条和状态显示
    progress_bar = st.progress(0)
    frame_text = st.empty()
    stats_text = st.empty()
    preview_placeholder = st.empty()
    
    # 用于统计检测结果
    detection_stats = {}
    preview_frames = []
    
    try:
        frame_count = 0
        processed_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 根据间隔处理帧
            if frame_count % frame_interval == 0:
                # 更新进度
                progress = int((processed_count / actual_total_frames) * 100)
                progress_bar.progress(progress)
                frame_text.text(f"处理帧: {processed_count}/{actual_total_frames} (原始帧: {frame_count}/{total_frames})")
                
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 运行检测
                results = model(frame_rgb, conf=confidence, iou=iou)
                
                # 绘制检测结果并统计
                for result in results:
                    # 更新统计信息
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        class_name = model.names[cls]
                        detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
                    
                    # 更新统计显示
                    stats_str = "检测统计：\n"
                    for cls_name, count in detection_stats.items():
                        stats_str += f"{cls_name}: {count}个\n"
                    stats_text.text(stats_str)
                    
                    # 绘制结果
                    plotted = result.plot()
                    
                    # 保存处理后的帧
                    plotted_bgr = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
                    out.write(plotted_bgr)
                    preview_out.write(plotted_bgr)
                    
                    # 每10帧更新一次预览
                    if processed_count % 10 == 0:
                        preview_placeholder.image(plotted, caption=f"实时预览 - 第{frame_count}帧", use_column_width=True)
                
                processed_count += 1
            
            frame_count += 1
            
        # 完成处理
        progress_bar.progress(100)
        frame_text.text("视频处理完成！")
        
    finally:
        # 释放资源
        cap.release()
        out.release()
        preview_out.release()
        
    return output_path, preview_path, detection_stats

def process_image(image):
    """处理图片，确保为RGB格式"""
    # 如果图片是RGBA格式，转换为RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # 如果是其他格式，也转换为RGB
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    return image

# 页面配置
st.set_page_config(
    page_title="YOLO v3物体检测演示",
    page_icon="🚗",
    layout="wide"
)

# 标题
st.title("YOLO v3实时物体检测演示")
st.markdown("这是一个使用YOLO v3进行实时物体检测的演示应用。")

# 侧边栏配置
st.sidebar.header("检测参数设置")

# 选择检测对象
detection_objects = st.sidebar.multiselect(
    "选择要检测的对象类型",
    ["行人", "自行车手", "汽车", "卡车", "交通信号灯"],
    default=["行人", "汽车"]
)

# 调整对象数量范围
min_objects = st.sidebar.number_input("最小对象数量", min_value=1, max_value=100, value=1)
max_objects = st.sidebar.number_input("最大对象数量", min_value=1, max_value=100, value=50)

# 模型参数调整
confidence_threshold = st.sidebar.slider(
    "置信度阈值",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "IOU阈值",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05
)

# 在侧边栏添加帧间隔设置
st.sidebar.markdown("---")
st.sidebar.header("视频处理设置")
frame_interval = st.sidebar.slider(
    "帧处理间隔（每N帧处理1帧）",
    min_value=1,
    max_value=30,
    value=5,
    help="值越大处理速度越快，但可能会错过一些画面。建议值：5-10"
)

# 初始化模型
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.info("请确保网络连接正常，首次运行时需要下载模型文件")
        return None

model = load_model()

if model is None:
    st.stop()

# 文件上传说明
st.markdown("""
### 文件上传说明
- 支持的图片格式：JPG, JPEG, PNG
- 支持的视频格式：MP4
- 视频文件大小限制：300MB
""")

# 文件上传
uploaded_file = st.file_uploader("选择图片或视频文件", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    try:
        # 获取文件大小（以MB为单位）
        file_size = uploaded_file.size / (1024 * 1024)
        
        # 如果是视频文件，检查大小限制
        if uploaded_file.type.startswith('video'):
            if file_size > 300:
                st.error(f"视频文件大小（{file_size:.1f}MB）超过限制（300MB）")
                st.stop()
            else:
                st.info(f"当前视频文件大小：{file_size:.1f}MB")
            
            try:
                # 保存上传的视频到临时文件
                temp_dir = tempfile.mkdtemp()
                temp_video_path = str(Path(temp_dir) / "input.mp4")
                
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 显示原始视频
                st.subheader("原始视频")
                st.video(uploaded_file)
                
                # 添加处理按钮
                if st.button("开始处理视频"):
                    st.subheader("处理进度")
                    
                    # 创建两列布局
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("实时预览")
                    
                    with col2:
                        st.subheader("检测统计")
                    
                    # 处理视频
                    output_path, preview_path, stats = process_video(
                        temp_video_path,
                        model,
                        confidence_threshold,
                        iou_threshold,
                        frame_interval
                    )
                    
                    # 显示处理后的视频
                    st.subheader("检测结果视频")
                    
                    # 创建下载按钮
                    with open(output_path, "rb") as f:
                        video_bytes = f.read()
                        st.download_button(
                            label="下载处理后的视频",
                            data=video_bytes,
                            file_name="detected_video.mp4",
                            mime="video/mp4"
                        )
                    
                    # 显示视频
                    st.video(video_bytes)
                    
                    # 显示总体统计结果
                    st.subheader("检测统计结果")
                    st.write(f"处理间隔：每{frame_interval}帧处理1帧")
                    for cls_name, count in stats.items():
                        st.write(f"- {cls_name}: 总计 {count}个")
                    
                    # 清理临时文件
                    try:
                        os.remove(temp_video_path)
                        os.remove(output_path)
                        os.remove(preview_path)
                        os.rmdir(temp_dir)
                    except Exception as e:
                        st.warning(f"清理临时文件时出错: {str(e)}")
                    
            except Exception as e:
                st.error(f"视频处理过程中出现错误: {str(e)}")
                st.info("请确保上传的是有效的MP4格式视频文件")
        
        # 处理图片
        elif uploaded_file.type.startswith('image'):
            # 读取并处理图片
            image = Image.open(uploaded_file)
            image = process_image(image)  # 确保图片为RGB格式
            image_np = np.array(image)
            
            # 显示原始图片
            st.subheader("原始图片")
            st.image(image, caption="上传的图片", use_column_width=True)
            
            with st.spinner('正在进行对象检测...'):
                # 运行检测
                results = model(image_np, conf=confidence_threshold, iou=iou_threshold)
                
                # 显示结果
                st.subheader("检测结果")
                for result in results:
                    plotted = result.plot()
                    st.image(plotted, caption="检测结果", use_column_width=True)
                    
                    # 显示检测到的对象统计
                    boxes = result.boxes
                    if len(boxes) > 0:
                        st.write("检测结果统计：")
                        for c in boxes.cls.unique():
                            n = (boxes.cls == c).sum()
                            class_name = model.names[int(c)]
                            st.write(f"- {class_name}: {n}个")
                    else:
                        st.info("未检测到任何对象")
            
    except Exception as e:
        st.error(f"处理过程中出现错误: {str(e)}")
        st.info("如果是图片格式问题，请尝试使用JPG或PNG格式的图片")

# 添加使用说明
with st.expander("使用说明"):
    st.markdown("""
    1. 在左侧边栏选择要检测的对象类型
    2. 调整对象数量范围
    3. 上传图片或视频文件
    4. 调整模型参数（置信度和重叠阈值）
    
    支持的对象类型：
    - 行人
    - 自行车手
    - 汽车
    - 卡车
    - 交通信号灯
    
    注意事项：
    - 首次运行时需要下载模型文件，请确保网络连接正常
    - 建议使用清晰的图片以获得更好的检测效果
    - 如果检测效果不理想，可以尝试调整置信度阈值
    - 视频文件大小不能超过300MB
    - 图片请使用RGB格式的JPG或PNG文件
    """)

# 添加页脚
st.markdown("---")
st.markdown("powered by YOLO v3 & Streamlit") 