# 导入flask类
from flask import Flask, request, jsonify, redirect, url_for


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
# 如果给路由传递参数，那么视图函数需要接受
@app.route('/diao/<user_id>')
def login(user_id):
    return '{}说老子tm直接求饶'.format(user_id)


# 指定请求方式
@app.route('/demo1',methods=['GET','POST'])
def demo1():
    # 直接从请求中得到请求方式
    return request.method


# 返回json
@app.route('/demo2')
def demo2():
    json_dict = {
        'user_id': 10,
        'user_name': 'jiebao'
    }

    return jsonify(json_dict)


# 重定向
@app.route('/demo3')
def demo3():
    return redirect('http://www.baidu.com')


# 重定向到自己的视图函数
@app.route('/demo4')
def demo4():
    return redirect(url_for('demo2'))


# 自定义状态码
@app.route('/demo5')
def demo5():
    return '状态码为123456',123456


# # 正则匹配路由：根据自己的规则去限定参数在进行访问
# from werkzeug.routing import BaseConverter
#
# # 自定义转换器
# class RegexConverter(BaseConverter):
#
#     def __init__(self, url_map, *args):
#         # super 重写父类
#         super(RegexConverter, self).__init__(url_map)
#         # 将第一个接受的参数当做匹配规则进行保存
#         self.regex = args[0]
#
#
# # 将自定义转换器添加到转换器字典中，并且指定使用时的名字：re
# app.url_map.converters['re'] = RegexConverter
#
# @app.route('/user/<re("[0-9]{3}"):user_id>')
# def index(user_id):
#     return user_id


if __name__=='__main__':
    # web服务器入口
    app.run()