# 使用
# python3 webstreaming.py --ip 0.0.0.0 --port 8000

# 导入必要的包
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# 初始化输出帧并使用一个锁来确保输出帧的线程
# 安全交换(对于多个浏览器/选项卡查看流非常有用)
outputFrame = None
lock = threading.Lock()

# 初始化flask对象
app = Flask(__name__)

# 初始化视频流并允许摄像机传感器预热
vs = VideoStream(0).start()
time.sleep(2.0)


@app.route("/")
def index():
    # 返回呈现的模板
    return render_template("index.html")


def detect_motion():
    # 获取对视频流、输出帧和锁定变量的全局引用
    global vs, outputFrame, lock

    # 从视频流循环帧
    while True:
        # 从视频流中读取下一帧，调整其大小，将帧转换为灰度，并使其模糊
        frame = vs.read()
        frame = imutils.resize(frame, width=640)

        # 获取当前时间戳并将其绘制到框架上
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%Y-%m-%d %H:%M:%S"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # 获取锁，设置输出帧，释放锁
        with lock:
            outputFrame = frame.copy()


def generate():
    # 获取输出帧的全局引用并锁定变量
    global outputFrame, lock

    # 循环输出流中的帧
    while True:
        # 等待，直到获得锁
        with lock:
            # 检查输出帧是否可用，否则跳过循环的迭代
            if outputFrame is None:
                continue

            # 将帧编码为JPEG格式
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # 确保帧已成功编码
            if not flag:
                continue

        # 以字节格式输出帧
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # 返回与特定媒体类型(mime类型)一起生成的响应
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# 检查这是否是执行的主线程
if __name__ == '__main__':
    # 构造参数解析器和解析命令行参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="设备的ip地址")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="服务器的临时端口号（1024到65535）")

    args = vars(ap.parse_args())

    # 启动一个线程
    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()

    # 启动flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# 释放视频流指针
vs.stop()
