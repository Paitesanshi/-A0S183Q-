from flask import Flask
from flask import render_template, redirect, url_for, request, session
import config
from flask_sqlalchemy import SQLAlchemy
import pymysql

app = Flask(__name__)
app.config.from_object(config)
db=SQLAlchemy(app)

@app.route('/china')
def china():
    return render_template('china.html')

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/infor')
def infor():
    return render_template('check.html')


# 登陆
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template('base.html')
    else:
        username = request.form.get('id')  # 与html页面名字相同
        password = request.form.get('password')
        print(username,password)
        user = User.query.filter(User.username == username,User.password==password).first()
        print(user)
        if user:
            session['username']=username
            session.permanent=True
            return redirect(url_for('china'))
        else:
            return '用户不存在'
# 注册
@app.route('/regis',methods=['GET','POST'])
def register():
    if request.method=='GET':

        return render_template('zhuce1.html')
    else:

        username=request.form.get('zcid')#与html页面名字相同
        password=request.form.get('zcpassword')
        user=User.query.filter(User.username==username).first()
        if user:
            return 'exit'
        else:
            user=User(username=username,password=password)
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('login'))

# 开始创建用户模型，即在数据库中创建表格：
class User(db.Model):
    __tablename__='user'
    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    username=db.Column(db.String(20),nullable=False)
    password=db.Column(db.String(20),nullable=False)

#创建表格
db.create_all()

#数据查询方法
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
    username=session.get('user')
    if username:
        return {'username':username}
    else:
        return {}


if __name__=='__main__':
    # web服务器入口
    app.run(debug=True)