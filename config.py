SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:数据库密码@数据库地址:3306/数据库名称?charset=utf8'
SQLALCHEMY_TRACK_MODIFICATIONS = False

import os
DEBUG = True

SECRET_KEY = os.urandom(24)

DIALECT = 'mysql'
DRIVER = 'mysqldb'
USERNAME = 'root'
PASSWORD = 'H9HbiPTwu)Oa'
HOST = '127.0.0.1'
PORT = '3306'
DATABASE = 'mis_db'