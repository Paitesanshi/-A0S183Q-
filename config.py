SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:zxc110@127.0.0.1:3306/test_db?charset=utf8'
SQLALCHEMY_TRACK_MODIFICATIONS = False

import os
DEBUG = True

SECRET_KEY = os.urandom(24)

DIALECT = 'mysql'
DRIVER = 'mysqldb'
USERNAME = 'root'
PASSWORD = 'zxc110'
HOST = '127.0.0.1'
PORT = '3306'
DATABASE = 'test_db'
