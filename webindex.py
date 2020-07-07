from flask import Flask
import pandas as pd
from flask import render_template, redirect, url_for, request, session
import config
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from flask_socketio import SocketIO
import json
import psutil
import pinyin.cedict

app = Flask(__name__)
app.config.from_object(config)
db = SQLAlchemy(app)
socketio = SocketIO(app)
name = str
data_raw = []


@app.route('/china')
def china():
    return render_template('china.html')


@app.route('/')
def index():
    return redirect(url_for('china'))


@app.route('/infor')
def infor():
    return render_template('check.html')


@socketio.on('name', namespace='/test')
def test_getarea(data):
    area = json.loads(data)
    print(area['area'])
    emit('server_response', "456")
    data_raw = pd.read_csv(area['area'] + "_min.csv")
    print(data_raw)


@socketio.on('date', namespace='/test')
def test_getdate(data):
    area = json.loads(data)
    print(area['date'])
    temp = {22, 33, 55, 44, 66, 77, 99}
    data_raw = pd.read_csv(area['area'] + "_min.csv")
    pos = data_raw[data_raw.date == area['date']].index
    temp = data_raw[pos:pos + 7]
    print(temp)
    emit('server_response', json.dumps(temp))


@socketio.on('connect', namespace='/test')
def test_connect():
    print("connected")
    emit('server_response', "123")


#
# @socketio.on('json', namespace='/test')
# def handle_json(json):
#     print('received json: ' + str(json))
# @socketio.on('message',namespace="/test")
# def handle_message(message):
#     print('received message:　' + message)

# 登陆
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('base.html')
    else:
        username = request.form.get('id')  # 与html页面名字相同
        password = request.form.get('password')
        print(username, password)
        user = User.query.filter(User.username == username, User.password == password).first()
        print(user)
        if user:
            session['username'] = username
            session.permanent = True
            return redirect(url_for('china'))
        else:
            return '用户不存在'


# 注册
@app.route('/regis', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':

        return render_template('zhuce1.html')
    else:

        username = request.form.get('zcid')  # 与html页面名字相同
        password = request.form.get('zcpassword')
        user = User.query.filter(User.username == username).first()
        if user:
            return 'exit'
        else:
            user = User(username=username, password=password)
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('login'))


# 开始创建用户模型，即在数据库中创建表格：
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(20), nullable=False)


# 创建表格
db.create_all()

# 数据查询方法
User.query.filter(User.username == 'mis1114').first()


# #数据添加方法
# user = User(username='wl097tql',password='g6666')
# db.session.add(user)
# db.session.commit()
#
# # 数据的修改方法
# user = User.query.filter(User.username=='wl097tql').first
# user.password='250250'
# db.session.commit()
#
# # 数据的删除方法
# user = User.query.filter(User.username=='wl097tql').first()
# db.session.delete(user)
# db.session.commit()

@app.context_processor
def mycontext():
    username = session.get('user')
    if username:
        return {'username': username}
    else:
        return {}


if __name__ == '__main__':
    # web服务器入口
    #app.run(debug=True)
    socketio.run(app, debug=True)
