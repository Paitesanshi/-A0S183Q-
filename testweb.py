from flask import Flask
from flask import render_template, redirect, url_for, request, session
import pymysql

from functools import wraps
from datetime import datetime

from werkzeug.security import generate_password_hash, check_password_hash  # 密码保护，使用hash方法

class Config(object):
    # 开启调试模式
    DEBUG = True

app = Flask(__name__)
app.config.from_object(Config)

class MySQLCommand(object):
    # 类的初始化
    def __init__(self):
        self.host = 'localhost'
        self.port = 3306  # 端口号
        self.user = 'root'  # 用户名
        self.password = ""  # 密码
        self.db = "home"  # 库
        self.table = "home_list"  # 表

    # 链接数据库
    def connectMysql(self):
        try:
            self.conn = pymysql.connect(host=self.host, port=self.port, user=self.user,
                                        passwd=self.password, db=self.db, charset='utf8')
            self.cursor = self.conn.cursor()
        except:
            print('connect mysql error.')

    # 查询数据
    def queryMysql(self):
        sql = "SELECT * FROM " + self.table

        try:
            self.cursor.execute(sql)
            row = self.cursor.fetchone()
            print(row)

        except:
            print(sql + ' execute failed.')

    # 插入数据
    def insertMysql(self, id, name, sex):
        sql = "INSERT INTO " + self.table + " VALUES(" + id + "," + "'" + name + "'," + "'" + sex + "')"
        try:
            self.cursor.execute(sql)
        except:
            print("insert failed.")

    # 更新数据
    def updateMysqlSN(self, name, sex):
        sql = "UPDATE " + self.table + " SET sex='" + sex + "'" + " WHERE name='" + name + "'"
        print("update sn:" + sql)

        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except:
            self.conn.rollback()

    def closeMysql(self):
        self.cursor.close()
        self.conn.close()


db = MySQLCommand()


#
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), nullable=False)
    _password = db.Column(db.String(200), nullable=False)  # 内部使用

    @property
    def password(self):  # 定义一个外部使用的密码
        return self._password

    @password.setter  # 设置密码加密
    def password(self, row_password):
        self._password = generate_password_hash(row_password)

    def check_password(self, row_password):  # 定义一个反向解密的函数
        result = check_password_hash(self._password, row_password)
        return result

#
# #
# class Question(db.Model):
#     __tablename__ = 'question'
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     author_id = db.Column(db.Integer, db.ForeignKey('user.id'))
#     title = db.Column(db.String(225), nullable=False)
#     detail = db.Column(db.Text, nullable=False)
#     classify = db.Column(db.Text, nullable=False)
#     time = db.Column(db.DateTime, default=datetime.now())
#     author = db.relationship('User', backref=db.backref('questions'))
#
#
# class Comment(db.Model):
#     __tablename__ = 'comment'
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     author_id = db.Column(db.Integer, db.ForeignKey('user.id'))
#     question_id = db.Column(db.Integer, db.ForeignKey('question.id'))
#     time = db.Column(db.DateTime, default=datetime.now())
#     detail = db.Column(db.Text, nullable=False)
#     question = db.relationship('Question', backref=db.backref('comments', order_by=time.desc))
#     author = db.relationship('User', backref=db.backref('comments'))


# 增加数据
# user = User(username='vae', password='5201314')
# db.session.add(user)
# db.session.commit()
# #
# # 查询数据
# user = User.query.filter(User.username == 'vae').first()
# print(user.username,user.password)
#
# #修改数据
# user.password = '250250'
# db.session.commit()

# db.create_all()




# # 将数据库查询结果传递到前端页面 Question.query.all(),问答排序
# @app.route('/')
# def index():
#     context = {
#         'questions': Question.query.order_by('-time').all()
#     }
#     return render_template('index.html', **context)
#

# 登录页面，用户将登录账号密码提交到数据库，如果数据库中存在该用户的用户名及id，返回首页
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        usern = request.form.get('username')
        passw = request.form.get('password')
        user = User.query.filter(User.username == usern).first()
        if user:
            if user.check_password(passw):
                session['user'] = usern
                session['id'] = user.id
                session.permanent = True
                return redirect(url_for('index'))  # 重定向到首页
            else:
                return u'password error'
        else:
            return u'username is not existed'


# 定义上下文处理器
@app.context_processor
def mycontext():
    usern = session.get('user')
    if usern:
        return {'username': usern}
    else:
        return {}


# 定义发布前登陆装饰器
def loginFrist(func):
    @wraps(func)
    def wrappers(*args, **kwargs):
        if session.get('user'):
            return func(*args, **kwargs)
        else:
            return redirect(url_for('login'))

    return wrappers


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter(User.username == username).first()
        if user:
            return 'username existed'
        else:
            user = User(username=username, password=password)
            db.session.add(user)  # 数据库操作
            db.session.commit()
            return redirect(url_for('login'))  # 重定向到登录页


# # 问答页面
# @app.route('/question', methods=['GET', 'POST'])
# @loginFrist
# def question():
#     if request.method == 'GET':
#         return render_template('question.html')
#     else:
#         title = request.form.get('title')
#         detail = request.form.get('detail')
#         classify = request.form.get('classify')
#         author_id = User.query.filter(User.username == session.get('user')).first().id
#         question = Question(title=title, detail=detail,classify=classify, author_id=author_id)
#         db.session.add(question)
#         db.session.commit()
#     return redirect(url_for('index'))  # 重定向到登录页


# @app.route('/detail/<question_id>')
# def detail(question_id):
#     quest = Question.query.filter(Question.id == question_id).first()
#     comments = Comment.query.filter(Comment.question_id == question_id).all()
#     return render_template('detail.html', ques=quest, comments=comments)


# # 读取前端页面数据，保存到数据库中
# @app.route('/comment/', methods=['POST'])
# @loginFrist
# def comment():
#     comment = request.form.get('new_comment')
#     ques_id = request.form.get('question_id')
#     auth_id = User.query.filter(User.username == session.get('user')).first().id
#     comm = Comment(author_id=auth_id, question_id=ques_id, detail=comment)
#     db.session.add(comm)
#     db.session.commit()
#     return redirect(url_for('detail', question_id=ques_id))


# 个人中心
@app.route('/usercenter/<user_id>/<tag>')
@loginFrist
def usercenter(user_id, tag):
    user = User.query.filter(User.id == user_id).first()
    context = {
        'user': user
    }
    if tag == '1':
        return render_template('usercenter1.html', **context)
    elif tag == '2':
        return render_template('usercenter2.html', **context)
    else:
        return render_template('usercenter3.html', **context)


# 修改密码
@app.route('/edit_password/', methods=['GET', 'POST'])
def edit_password():
    if request.method == 'GET':
        return render_template("edit_password.html")
    else:
        newpassword = request.form.get('password')
        user = User.query.filter(User.id == session.get('id')).first()
        user.password = newpassword
        db.session.commit()
        return redirect(url_for('index'))


# 等待
@app.route('/wait')
def wait():
    if request.method == 'GET':
        return render_template("wait.html")


if __name__ == '__main__':
    app.run(debug=True)