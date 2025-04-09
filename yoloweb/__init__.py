import pymysql

# 驱动伪装
pymysql.install_as_MySQLdb()
# python manage.py migrate  根据sql生成model