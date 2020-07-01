#导入flask类
from flask import Flask


# 配置对象 里面定义需要给app添加的一系列配置
class Config(object):
    # 开启调试模式
    DEBUG = True


# 创建一个app应用
# __nanme__指向程序所在的包
# 初始化参数 import_name Flask程序所在的包
# static_url_path 静态文件访问路径 可以不传 默认使用：static_folder
# static_folder 静态文件储存的文件夹 可以不传 默认使用：static
# template_folder 模板文件储存文件夹 默认templates
app = Flask(__name__)
# 从配置对象加载配置
app.config.from_object(Config)


# 装饰器的作用：将路由映射到视图函数
@app.route('/')
def index():
    return '磊哥yyds 没有任何说头'
@app.route('/diao/')
def login():
    return '老子tm直接求饶'


if __name__=='__main__':
    # web服务器入口
    app.run()