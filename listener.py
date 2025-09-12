import threading
from flask import Flask, jsonify
from cyber.python.cyber_py3 import cyber
from modules.common_msgs.localization_msgs.localization_pb2 import LocalizationEstimate

# 定义全局变量用于存储最新定位数据
latest_localization = {}

# 用于记录上一次的 x、y 坐标
last_x = None
last_y = None

# 在全局创建 Flask 应用
app = Flask(__name__)

# 在这里添加 after_request，禁用缓存
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/var', methods=['GET'])
def get_variable():
    """
    返回最新定位数据，格式为 JSON。
    如果还没有数据，则返回空字典。
    """
    return jsonify(latest_localization)

def run_flask():
    """
    启动 Flask 服务器，监听 127.0.0.1:5000
    """
    app.run(host='127.0.0.1', port=5000)

def localization_callback(data):
    """
    定位数据回调函数：仅当 x 或 y 与上一次相比变化超过 5 时才更新 latest_localization
    """
    global latest_localization, last_x, last_y

    current_x = data.pose.position.x
    current_y = data.pose.position.y

    # 如果是第一次接收数据（last_x、last_y 还未初始化），先初始化
    if last_x is None or last_y is None:
        last_x = current_x
        last_y = current_y
        # 第一次可选择是否立即“发送”，这里直接更新并打印
        latest_localization = {
            'timestamp': data.header.timestamp_sec,
            'position': {
                'x': current_x,
                'y': current_y,
                'z': data.pose.position.z
            },
            'orientation': {
                'qw': data.pose.orientation.qw,
                'qx': data.pose.orientation.qx,
                'qy': data.pose.orientation.qy,
                'qz': data.pose.orientation.qz
            }
        }
        print("=" * 80)
        print("Received first localization data (automatically sent):")
        print(f"Timestamp: {data.header.timestamp_sec}")
        print(f"Position: x={current_x}, y={current_y}, z={data.pose.position.z}")
        print(f"Orientation: qw={data.pose.orientation.qw}, qx={data.pose.orientation.qx}, "
              f"qy={data.pose.orientation.qy}, qz={data.pose.orientation.qz}")
        print("=" * 80)
    else:
        # 如果 x 或 y 改变超过 5，则更新并输出
        if abs(current_x - last_x) > 5 or abs(current_y - last_y) > 5:
            last_x = current_x
            last_y = current_y

            latest_localization = {
                'timestamp': data.header.timestamp_sec,
                'position': {
                    'x': current_x,
                    'y': current_y,
                    'z': data.pose.position.z
                },
                'orientation': {
                    'qw': data.pose.orientation.qw,
                    'qx': data.pose.orientation.qx,
                    'qy': data.pose.orientation.qy,
                    'qz': data.pose.orientation.qz
                }
            }

            print("=" * 80)
            print("Localization changed by more than 5 (x or y), sending data:")
            print(f"Timestamp: {data.header.timestamp_sec}")
            print(f"Position: x={current_x}, y={current_y}, z={data.pose.position.z}")
            print(f"Orientation: qw={data.pose.orientation.qw}, qx={data.pose.orientation.qx}, "
                  f"qy={data.pose.orientation.qy}, qz={data.pose.orientation.qz}")
            print("=" * 80)
        else:
            # 如果变化不超过 5，不做任何更新
            pass

def localization_listener():
    """
    创建定位模块的 Reader，并启动消息接收
    """
    print("=" * 120)
    node = cyber.Node("localization_listener")
    node.create_reader("/apollo/localization/pose", LocalizationEstimate, localization_callback)
    node.spin()

if __name__ == '__main__':
    cyber.init()

    # 在独立线程中启动 Flask 服务器
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.setDaemon(True)
    flask_thread.start()

    # 启动定位数据的监听
    localization_listener()
    cyber.shutdown()
