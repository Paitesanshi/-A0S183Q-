from flask import Flask
import pandas as pd
from flask import render_template, redirect, url_for, request, session
import config
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
import json
from datetime import timedelta
import pymysql
import pinyin.cedict

app = Flask(__name__)
app.config.from_object(config)
app.config['SECRET_KEY'] = 'XX77XXX'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
db = SQLAlchemy(app)
socketio = SocketIO(app)
name = ''

# 开始创建用户模型，即在数据库中创建表格：
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(20), nullable=False)
    kind = db.Column(db.String(20), nullable=False)
    def to_json(model):
        """ Returns a JSON representation of an SQLAlchemy-backed object. """
        json = {}
        # json['fields'] = {}
        # json['pk'] = getattr(model, 'id')
        for col in model._sa_class_manager.mapper.mapped_table.columns:
            # json['fields'][col.name] = getattr(model, col.name)
            json[col.name] = getattr(model, col.name)
        # return dumps([json])
        return json
    def to_json_list(model_list,self):
        json_list = []
        for model in model_list:
            json_list.append(self.to_json(model))
        return json_list

# 创建表格
db.create_all()
@app.route('/china')
def china():
    return render_template('china.html')


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/infor')
def infor():
    return render_template('check.html')

@socketio.on('name', namespace='/test')
def test_getarea(data):
    global name
    area = json.loads(data)
    name = area['area']

@socketio.on('getusers', namespace='/test')
def test_getusers(data):
    users=User.query.filter(User.kind == 'user').all()
    result = User.to_json_list(users,User)
    print(result)
    emit('users',json.dumps(result))

@socketio.on('deleteuser', namespace='/test')
def test_deleteuser(data):
    username=json.loads(data)
    user=User.query.filter(User.username == username).first()
    print(user)
    db.session.delete(user)
    db.session.commit()

@socketio.on('register', namespace='/test')
def test_register(data):
    data = json.loads(data)
    username = data['username']  # 与html页面名字相同
    password = data['password']
    kind=data['kind']
    user = User.query.filter(User.username == username).first()
    if user:
        emit('register_response', json.dumps("exist"))
    else:
        user = User(username=username, password=password,kind=kind)
        db.session.add(user)
        db.session.commit()
        emit('register_response', json.dumps("ok"))

@socketio.on('mylogin', namespace='/test')
def test_login(data):
    data = json.loads(data)
    username = data['username']  # 与html页面名字相同
    password = data['password']
    kind=data['kind']
    user = User.query.filter(User.username == username, User.password == password,User.kind==kind).first()
    if user:
        session['username'] = username
        session.permanent = True
        # # return redirect(url_for('china'))
        # return render_template('china.html')
        emit('login_response', json.dumps("ok"))
    else:
        emit('login_response', json.dumps("not exist"))


@socketio.on('date', namespace='/test')
def test_getdate(data):
    area = json.loads(data)
    data_raw = pd.read_csv(name + "_max.csv")
    pos = data_raw[data_raw['date'] == area['date']].index.to_list()
    temp_max = data_raw['tmax'][pos[0]:pos[0] + 7]
    temp_max = temp_max.to_list()
    emit('server_response_max', json.dumps(temp_max))
    data_raw = pd.read_csv(name + "_min.csv")
    pos = data_raw[data_raw['date'] == area['date']].index.to_list()
    temp_min = data_raw['tmin'][pos[0]:pos[0] + 7]
    temp_min = temp_min.to_list()
    emit('server_response_min', json.dumps(temp_min))
    # emit('server_response', pd.Series(temp).to_json(orient='values'))


@socketio.on('connect', namespace='/test')
def test_connect():
    print("connected")
    # users = User.query.filter(User.kind == 'user').all()
    # print(users[0].username)
    # print(json.dumps(users[0].username))

#
# @socketio.on('json', namespace='/test')
# def handle_json(json):
#     print('received json: ' + str(json))
# @socketio.on('message',namespace="/test")
# def handle_message(message):
#     print('received message:　' + message)

# 登陆
@app.route('/login')
def login():
    return render_template('user.html')


# 注册
@app.route('/regis')
def register():
    return render_template('zhuce1.html')





# 数据查询方法

#
# # 数据添加方法
# user = User(username='wl097tql', password='g6666')
# db.session.add(user)
# db.session.commit()
#
# # 数据的修改方法
# user = User.query.filter(User.username == 'wl097tql').first
#
# db.session.commit()
#
# # 数据的删除方法
# user = User.query.filter(User.username == 'wl097tql').first()
# db.session.delete(user)
# db.session.commit()


@app.context_processor
def mycontext():
    username = session.get('user')
    if username:
        return {'username': username}
    else:
        return {}

#
if __name__ == '__main__':
    # web服务器入口
    socketio.run(app,debug=True)
