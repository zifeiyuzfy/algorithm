import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
	model = YOLO('ultralytics/cfg/models/11/yolo11s.yaml')
	# model.load('yolo11s.pt')  #加载预训练权重
	model.load('runs/detect/train5/weights/best.pt')  # 加载预训练权重ssss
	model.train(data='People.yaml',   #数据集yaml文件
				imgsz=640,  # 输入图像尺寸
				epochs=100,  # 训练轮数
				batch=8,  # 每个批次的图像数量
				workers=8,  # 数据加载的线程数
				device=0,  # 使用的设备（0 表示 GPU，'cpu' 表示 CPU）
				optimizer='SGD',  # 优化器类型
				patience=50 ,  # 在训练时，如果经过20轮性能没有提升，则停止训练（早停机制）
				amp=True,  # 是否使用自动混合精度训练
				cache=False,  # 是否缓存数据集到内存
	)