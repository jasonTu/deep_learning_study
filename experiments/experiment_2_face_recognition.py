"""
实验二：人脸识别实战
使用少量照片训练，实现实时人脸识别
"""

import os
import cv2
import numpy as np
from PIL import Image
import pickle
from datetime import datetime

# 尝试导入深度学习框架
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    from torch.utils.data import Dataset, DataLoader
    USE_PYTORCH = True
    print("使用 PyTorch 框架")
except ImportError:
    USE_PYTORCH = False
    print("PyTorch 未安装，使用 OpenCV DNN 模块")

# ============================================
# 配置
# ============================================
DATA_DIR = "/Users/tuyouwu/sharing/deep_learning/data/face_dataset"
MODEL_PATH = "/Users/tuyouwu/sharing/deep_learning/data/face_recognition_model.pkl"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================
# 1. 数据收集模块
# ============================================
def collect_face_data(person_name, num_samples=20):
    """
    收集人脸数据
    
    参数:
    person_name -- 人员姓名（用作标签和文件夹名）
    num_samples -- 收集样本数量
    """
    person_dir = os.path.join(DATA_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    count = 0
    print(f"开始收集 {person_name} 的人脸数据")
    print("按 'c' 捕获照片，按 'q' 退出")
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # 绘制人脸框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Collect Face Data - Press c to capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces) > 0:
            # 保存人脸区域
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                
                filename = f"{person_name}_{count:03d}.jpg"
                filepath = os.path.join(person_dir, filename)
                cv2.imwrite(filepath, face_img)
                
                count += 1
                print(f"已保存: {filename} ({count}/{num_samples})")
                break
                
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n完成！共收集 {count} 张照片，保存在 {person_dir}")

# ============================================
# 2. 数据增强
# ============================================
def augment_image(image):
    """对图像进行数据增强"""
    augmented = []
    
    # 原图
    augmented.append(image)
    
    # 水平翻转
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)
    
    # 轻微旋转
    rows, cols = image.shape
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented.append(rotated)
    
    # 调整亮度
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
    augmented.append(bright)
    augmented.append(dark)
    
    return augmented

def prepare_training_data():
    """准备训练数据"""
    faces = []
    labels = []
    label_map = {}
    current_label = 0
    
    print("准备训练数据...")
    
    for person_name in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        label_map[current_label] = person_name
        print(f"处理: {person_name} (标签: {current_label})")
        
        for img_name in os.listdir(person_dir):
            if not img_name.endswith('.jpg'):
                continue
            
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # 数据增强
                augmented_imgs = augment_image(img)
                for aug_img in augmented_imgs:
                    faces.append(aug_img)
                    labels.append(current_label)
        
        current_label += 1
    
    print(f"\n总共 {len(faces)} 张训练图像")
    print(f"类别数量: {len(label_map)}")
    
    return faces, np.array(labels), label_map

# ============================================
# 3. 训练模型 (OpenCV LBPH)
# ============================================
def train_model_opencv():
    """使用 OpenCV 的 LBPH 人脸识别器训练"""
    faces, labels, label_map = prepare_training_data()
    
    if len(faces) == 0:
        print("没有训练数据！请先收集人脸数据")
        return None, None
    
    # 创建 LBPH 识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    
    print("\n开始训练...")
    recognizer.train(faces, labels)
    
    # 保存模型
    recognizer.save(MODEL_PATH)
    with open(MODEL_PATH.replace('.pkl', '_labels.pkl'), 'wb') as f:
        pickle.dump(label_map, f)
    
    print(f"模型已保存到: {MODEL_PATH}")
    
    return recognizer, label_map

# ============================================
# 4. 实时识别
# ============================================
def recognize_faces():
    """实时人脸识别"""
    # 加载模型
    if not os.path.exists(MODEL_PATH):
        print("模型不存在！请先训练")
        return
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    
    with open(MODEL_PATH.replace('.pkl', '_labels.pkl'), 'rb') as f:
        label_map = pickle.load(f)
    
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("\n实时识别中...")
    print("按 'q' 退出，按 's' 截图保存")
    
    confidence_threshold = 100  # 置信度阈值，越小越严格
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # 预测
            label_id, confidence = recognizer.predict(face_roi)
            
            # 显示结果
            if confidence < confidence_threshold:
                name = label_map.get(label_id, "Unknown")
                color = (0, 255, 0)  # 绿色
                text = f"{name} ({confidence:.1f})"
            else:
                name = "Unknown"
                color = (0, 0, 255)  # 红色
                text = f"Unknown ({confidence:.1f})"
            
            # 绘制
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 显示帧率
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/Users/tuyouwu/sharing/deep_learning/data/screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"截图已保存: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("识别结束")

# ============================================
# 5. 单张图片识别
# ============================================
def recognize_single_image(image_path):
    """识别单张图片中的人脸"""
    if not os.path.exists(MODEL_PATH):
        print("模型不存在！请先训练")
        return
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    
    with open(MODEL_PATH.replace('.pkl', '_labels.pkl'), 'rb') as f:
        label_map = pickle.load(f)
    
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图片: {image_path}")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    print(f"检测到 {len(faces)} 个人脸")
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        
        label_id, confidence = recognizer.predict(face_roi)
        
        if confidence < 100:
            name = label_map.get(label_id, "Unknown")
            print(f"识别结果: {name}, 置信度: {confidence:.2f}")
        else:
            print(f"识别结果: Unknown, 置信度: {confidence:.2f}")
        
        # 绘制结果
        color = (0, 255, 0) if confidence < 100 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # 保存结果
    result_path = image_path.replace('.', '_result.')
    cv2.imwrite(result_path, frame)
    print(f"结果已保存: {result_path}")
    
    # 显示
    cv2.imshow('Recognition Result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ============================================
# 6. 菜单界面
# ============================================
def show_menu():
    """显示菜单"""
    print("\n" + "=" * 50)
    print("        人脸识别实验系统")
    print("=" * 50)
    print("1. 收集人脸数据")
    print("2. 训练模型")
    print("3. 实时识别")
    print("4. 识别单张图片")
    print("5. 退出")
    print("=" * 50)

# ============================================
# 主程序
# ============================================
def main():
    print("=" * 50)
    print("实验二：人脸识别实战")
    print("=" * 50)
    
    while True:
        show_menu()
        choice = input("请选择操作 (1-5): ").strip()
        
        if choice == '1':
            name = input("请输入人员姓名: ").strip()
            num = int(input("收集样本数量 (默认20): ") or "20")
            collect_face_data(name, num)
            
        elif choice == '2':
            train_model_opencv()
            
        elif choice == '3':
            recognize_faces()
            
        elif choice == '4':
            path = input("请输入图片路径: ").strip()
            recognize_single_image(path)
            
        elif choice == '5':
            print("再见！")
            break
            
        else:
            print("无效选择，请重试")

if __name__ == "__main__":
    # 检查 OpenCV contrib 模块
    try:
        test = cv2.face.LBPHFaceRecognizer_create()
        print("OpenCV face 模块已就绪")
    except AttributeError:
        print("错误: 需要安装 opencv-contrib-python")
        print("运行: pip install opencv-contrib-python")
        exit(1)
    
    main()
