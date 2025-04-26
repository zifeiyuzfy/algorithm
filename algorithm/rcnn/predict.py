#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from frcnn import FRCNN

# 定义全局变量 frcnn
# frcnn = FRCNN()
# 启用Soft-NMS
frcnn = FRCNN()

def is_target_detected(frame):
    """
    判断帧中是否检测到目标
    :param frame: 输入的帧
    :return: 是否检测到目标的布尔值
    """
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    _, _, classes = frcnn.detect_image(frame, return_classes=True)
    return len(classes) > 0

def get_detected_target_types(frame):
    """
    获取检测到的目标的类型
    :param frame: 输入的帧
    :return: 检测到的目标类型列表
    """
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    _, _, classes = frcnn.detect_image(frame, return_classes=True)
    return classes


if __name__ == "__main__":
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "video"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop = False
    count = True
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    # video_path = 0  # 使用摄像头进行检测
    video_path = "video/people2.mp4"  # 使用摄像头进行检测
    video_save_path = "video_out/people2_out.mp4"  # 修改为具体的视频文件名
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/people.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    # 当模式为单张图片预测时
    if mode == "predict":
        '''
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_dr_txt.py，在get_dr_txt.py即实现了遍历还实现了目标信息的保存。
        2、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        3、如果想要获得预测框的坐标，可以进入frcnn.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        4、如果想要利用预测框截取下目标，可以进入frcnn.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        5、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入frcnn.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        # 进入一个无限循环，持续等待用户输入图片文件名进行预测
        while True:
            # 提示用户输入要进行预测的图片文件名
            img = input('Input image filename:')
            try:
                # 尝试使用PIL库的Image.open方法打开用户输入的图片文件
                # 如果文件不存在、格式不支持或其他读取错误，会抛出异常
                image = Image.open(img)
            except:
                # 如果打开图片文件时出现异常，打印错误提示信息
                print('Open Error! Try again!')
                # 跳过本次循环，继续等待用户输入新的文件名
                continue
            else:
                # 如果成功打开图片文件，调用frcnn实例的detect_image方法进行目标检测
                # crop参数指定是否在检测后对目标进行截取，count参数指定是否进行目标计数
                r_image = frcnn.detect_image(image, crop=crop, count=count)
                # 使用PIL库的show方法显示检测后的图片
                r_image.show()

    # 当模式为视频检测时
    elif mode == "video":
        # 使用OpenCV的VideoCapture类打开指定的视频源
        # 如果video_path为0，表示使用默认摄像头；如果是文件路径，则打开对应的视频文件
        capture = cv2.VideoCapture(video_path)

        # 检查摄像头或视频文件是否成功打开
        if not capture.isOpened():
            # 如果打开失败，打印错误信息，包含视频源路径
            print(f"Error opening camera: {video_path}")
            # 终止程序运行
            exit()

        # 获取原始视频的帧率
        video_fps = capture.get(cv2.CAP_PROP_FPS)

        # 如果指定了视频保存路径
        if video_save_path != "":
            # 定义视频编码格式为mp4v，用于将处理后的视频帧保存为MP4文件
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 获取视频的宽度和高度，用于创建VideoWriter对象
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # 创建VideoWriter对象，用于将处理后的视频帧写入指定路径的文件
            # 参数依次为：保存路径、编码格式、帧率、视频尺寸
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        # 使用OpenCV创建一个可调整大小的窗口，窗口名称为"video"
        cv2.namedWindow("video", cv2.WINDOW_NORMAL)
        # 将窗口的初始大小设置为640x480像素
        cv2.resizeWindow("video", 640, 480)

        # 初始化帧率变量，用于记录视频处理的帧率
        fps = 0.0
        try:
            # 进入一个无限循环，逐帧读取视频并进行处理
            while True:
                # 记录当前时间，用于计算处理一帧的时间
                t1 = time.time()
                # 从视频捕获对象中读取一帧
                # ref是一个布尔值，表示是否成功读取到帧；frame是读取到的帧图像
                ref, frame = capture.read()

                # 检查是否成功读取到帧
                if not ref:
                    # 跳出循环，结束视频处理
                    break

                # 判断是否检测到目标
                detected = is_target_detected(frame)
                if detected:
                    target_types = get_detected_target_types(frame)
                    print("检测到的目标类型:", target_types)

                # OpenCV读取的图像是BGR格式，而模型处理通常需要RGB格式
                # 因此将BGR格式的帧图像转换为RGB格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 将NumPy数组表示的图像转换为PIL的Image对象
                # 以便传入frcnn模型的detect_image方法进行处理
                frame = Image.fromarray(np.uint8(frame))
                # 调用frcnn实例的detect_image方法对图像进行目标检测
                # 并将检测结果转换为NumPy数组
                frame = np.array(frcnn.detect_image(frame))
                # 将处理后的RGB格式图像转换回BGR格式
                # 以便使用OpenCV进行显示和保存
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 计算当前的帧率
                # 使用指数加权平均的方法更新帧率值，使帧率计算更平滑
                fps = (fps + (1. / (time.time() - t1))) / 2
                # 打印当前的帧率，保留两位小数
                print("fps= %.2f" % (fps))
                # 在图像上绘制当前的帧率信息
                # 参数依次为：图像、文本内容、文本位置、字体、字体大小、文本颜色、文本厚度、线条类型（抗锯齿）
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # 在创建的窗口中显示处理后的帧图像
                cv2.imshow("video", frame)
                # 等待1毫秒，检查是否有按键输入
                # cv2.waitKey(1)返回按键的ASCII码值，与0xff进行按位与操作，确保结果在0-255范围内
                c = cv2.waitKey(1) & 0xff

                # 如果指定了视频保存路径，将处理后的帧写入VideoWriter对象进行保存
                if video_save_path != "":
                    out.write(frame)

                # 检查是否按下了 'q' 键
                # ord('q')返回字符 'q' 的ASCII码值
                if c == ord('q'):
                    # 如果按下 'q' 键，打印提示信息，表示正在终止程序并保存视频
                    print("Terminating and saving video...")
                    # 跳出循环，结束视频处理
                    break
        except KeyboardInterrupt:
            # 如果用户按下Ctrl + C，捕获KeyboardInterrupt异常
            # 打印提示信息，表示检测到键盘中断，正在终止程序并保存视频
            print("KeyboardInterrupt detected. Terminating and saving video...")
        finally:
            # 无论是否发生异常，都会执行finally块中的代码
            # 释放视频捕获对象，关闭摄像头或视频文件
            capture.release()
            # 如果指定了视频保存路径，释放VideoWriter对象，完成视频保存
            if video_save_path != "":
                out.release()
            # 关闭所有OpenCV打开的窗口
            cv2.destroyAllWindows()

    # 当模式为FPS测试时
    elif mode == "fps":
        # 使用PIL库的Image.open方法打开指定的用于FPS测试的图片文件
        img = Image.open(fps_image_path)
        # 调用frcnn实例的get_FPS方法，传入图片和检测次数
        # 计算处理图片所需的时间
        tact_time = frcnn.get_FPS(img, test_interval)
        # 打印处理图片所需的时间和对应的FPS值
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    # 当模式为文件夹批量图片预测时
    elif mode == "dir_predict":
        # 导入os模块，用于文件和目录操作
        import os
        # 导入tqdm模块，用于显示进度条
        from tqdm import tqdm

        # 获取指定文件夹中的所有文件名
        img_names = os.listdir(dir_origin_path)
        # 使用tqdm显示进度条，遍历文件夹中的所有文件名
        for img_name in tqdm(img_names):
            # 检查文件名是否以常见的图像文件扩展名结尾
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                # 构建完整的图像文件路径
                image_path = os.path.join(dir_origin_path, img_name)
                # 使用PIL库的Image.open方法打开图像文件
                image = Image.open(image_path)
                # 调用frcnn实例的detect_image方法对图像进行目标检测
                r_image = frcnn.detect_image(image)
                # 检查保存结果的文件夹是否存在，如果不存在则创建该文件夹
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                # 将检测后的图像保存到指定的文件夹中
                # 将原文件扩展名替换为.png，同时设置图片质量和子采样参数
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    # 如果模式不是上述四种有效模式之一
    else:
        # 抛出断言错误，提示用户指定正确的模式
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")