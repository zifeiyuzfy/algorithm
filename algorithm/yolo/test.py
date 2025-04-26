from ultralytics import YOLO
import os
import cv2
import numpy as np

# 加载训练好的模型，改为自己的路径
model = YOLO('runs/detect/train5/weights/best.pt')  # 修改为训练好的路径
image_paths = [
    # 'img/people.jpg',
    'img/people3.jpg'
]
output_folder = 'imgout'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for source in image_paths:
    class_count = {0: 0, 1: 0}  # 初始化 0 类和 1 类的计数
    results = model.predict(source, conf=0.05)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            if cls in [0, 1]:
                class_count[cls] = class_count.get(cls, 0) + 1

    total_count = class_count[0] + class_count[1]

    # 保存检测结果
    filename = os.path.basename(source)
    new_filename = f"test_{filename}"
    output_path = os.path.join(output_folder, new_filename)
    result.save(output_path)

    # 在输出图片上添加总人数信息
    img = cv2.imread(output_path)
    text = f"Total people:{total_count}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)

    # 输出每个类别的数量
    for cls, count in class_count.items():
        print(f"Class {cls}: {count} instances")
