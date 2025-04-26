import cv2
import ffmpeg
import numpy as np

# 视频流地址
input_url = 'rtmp://localhost:1935/flv-live/test'
output_url = 'rtmp://localhost:1935/flv-live/test02'
# 打开RTMP输入流
cap = cv2.VideoCapture(input_url)
if not cap.isOpened():
    print("无法打开视频流")
    exit()

# 获取视频的基本信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps)
print(width)
print(height)


# 使用 FFmpeg 启动推流进程（保持进程持续运行）
process = (
    ffmpeg
    .input('pipe:0', format="rawvideo", pix_fmt="bgr24", s=f'{width}x{height}')  # 从标准输入流读取帧
    .output(output_url, vcodec='libx264', pix_fmt='yuv420p', r=fps, s=f'{width}x{height}', format='flv')  # 输出到 RTMP 流
    .run_async(pipe_stdin=True)  # 异步执行 FFmpeg，允许我们向标准输入写入数据
)



print("启动")
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧，退出")
        break

    # 在每一帧的左上角绘制一个红色矩形框
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    # face_detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml");
    # faces = face_detector.detectMultiScale(gray, 1.1, 5);
    # for x, y, w, h in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2);


    # 将处理后的帧传给 FFmpeg 进行推流
    process.stdin.write(frame.tobytes())  # 写入 FFmpeg 标准输入流

    # 显示当前帧（可选）
    cv2.imshow('Stream', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
