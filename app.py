# encoding: utf-8


# 导入Flask类

from flask import Flask

# 创建Flask实例

app = Flask(__name__)


# 定义路由, 由index函数处理url为/的GET请求

@app.route('/')
def index():
    # 返回响应内容

    return 'Hello, Silence'


# 脚本运行执行代码

if __name__ == '__main__':
    # 启动Flask实例, 设置监听0.0.0.0:9001, 开启调试模式
    app.run()
