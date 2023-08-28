import pymysql
from prettytable import PrettyTable
import gradio as gr

def sql_result_to_table_str(sql_result):
    # Create a PrettyTable object
    table = PrettyTable()
    table.field_names = sql_result[0].keys()

    # Add rows to the table
    for row in sql_result:
        table.add_row(row.values())
    table.float_format = ".2"
    return str(table)


class MySQLDB(object):
    def __init__(self):
        self.default_db = ["information_schema", "mysql", "performance_schema", "sys"]
        self.host = None
        self.user = None
        self.password = None
        self.port = None
        self.database = None
        self.conn = None
        self.cursor = None


    def connect(self, host, user, password, port, database=None):
        if host == '' or user == '' or password == '' or port == '':
            raise gr.Error('请填写完整的数据库连接信息')
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.database = database
        self.disconnect()
        try:
            self.conn = pymysql.connect(
                host=host,
                port=int(port),
                user=user,
                password=password,
                database=database,
                cursorclass=pymysql.cursors.DictCursor
            )
            self.cursor = self.conn.cursor()
        except Exception as e:
            error_message = f"SQL error: {str(e)}"
            raise gr.Error(error_message)

    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def execute_sql(self, sql, raise_err=False):
        try:
            for sub_sql in sql.split(";"):
                sub_sql = sub_sql.strip()
                if len(sub_sql) > 0:
                    self.cursor.execute(sub_sql)
            result = self.cursor.fetchall()
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            error_message = f"SQL error: {str(e)}"
            if raise_err:
                raise e
            else:
                return e, error_message

        # print(result)
        # convert query result to string
        if result:
            # rows = [', '.join(str(v) for v in row.values()) for row in result]
            # header = ', '.join(result[0].keys())
            # out_str = f"{header}\n{'-' * len(header)}\n" + '\n'.join(rows)
            out_str = sql_result_to_table_str(result)
        else:
            if "create" in sql.lower():
                out_str = "create table successfully."
            elif "insert" in sql.lower():
                out_str = "insert data successfully."
            elif "delete" in sql.lower():
                out_str = "delete data successfully."
            elif "update" in sql.lower():
                out_str = "update data successfully."
            elif 'alter' in sql.lower():
                out_str = "alter table successfully."
            elif 'drop' in sql.lower():
                out_str = "drop table successfully."
            else:
                out_str = "no results found."

        return result, out_str

    def select(self, table, columns="*", condition=None):
        sql = f"SELECT {columns} FROM {table}"
        if condition:
            sql += f" WHERE {condition}"
        return self.execute_sql(sql)

    def insert(self, table, data):
        keys = ','.join(data.keys())
        values = ','.join([f"'{v}'" for v in data.values()])
        sql = f"INSERT INTO {table} ({keys}) VALUES ({values})"
        return self.execute_sql(sql)

    def update(self, table, data, condition):
        set_values = ','.join([f"{k}='{v}'" for k, v in data.items()])
        sql = f"UPDATE {table} SET {set_values} WHERE {condition}"
        return self.execute_sql(sql)

    def delete(self, table, condition):
        sql = f"DELETE FROM {table} WHERE {condition}"
        return self.execute_sql(sql)

    def create_database(self, database):
        try:
            self.execute_sql(f"DROP DATABASE `{database}`", raise_err=True)
        except Exception as e:
            pass
        sql = f"CREATE DATABASE `{database}`"
        self.execute_sql(sql, raise_err=True)
        # if self.database is None:
        #     self.database = database
        return True

    def drop_database(self, ):
        assert self.database is not None
        sql = f"DROP DATABASE `{self.database}`"
        self.execute_sql(sql, raise_err=True)
        self.database = None

    def get_table_details(self):
        if self.database is None:
            return None
        # 生成数据库中所有表的创建语句
        sql = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = '{self.database}'"
        result, _ = self.execute_sql(sql)
        table_names = [row["TABLE_NAME"] for row in result]
        table_details = []
        for table_name in table_names:
            sql = f"SHOW CREATE TABLE `{table_name}`"
            result, _ = self.execute_sql(sql)
            table_details.append(result[0]["Create Table"])
        # 去除符号``
        table_details = [table.replace("`", "") for table in table_details]
        # 去除类似字符串ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        table_details = [table.split(" ENGINE")[0]+';' for table in table_details]
        # 将列表转换为字符串，以\n分隔
        table_details = "\n".join(table_details)
        return table_details
    
    def get_databases(self):
        sql = "SHOW DATABASES;"
        result, _ = self.execute_sql(sql)
        databases = [row["Database"] for row in result if row["Database"] not in self.default_db]
        return databases
    
    def get_tables(self,database):
        sql = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = '{database}'"
        result, _ = self.execute_sql(sql)
        tables = [row["TABLE_NAME"] for row in result]
        return tables
    
    def get_table_data(self,table):
        sql = f"SELECT * FROM `{table}` LIMIT 100;"
        try:
            for sub_sql in sql.split(";"):
                sub_sql = sub_sql.strip()
                if len(sub_sql) > 0:
                    self.cursor.execute(sub_sql)
            result = self.cursor.fetchall()
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
        return result
    
    def get_fields(self,table):
        # 获取表的字段
        sql = f"SHOW COLUMNS FROM `{table}`;"
        try:
            for sub_sql in sql.split(";"):
                sub_sql = sub_sql.strip()
                if len(sub_sql) > 0:
                    self.cursor.execute(sub_sql)
            result = self.cursor.fetchall()
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
        fields = [row["Field"] for row in result]
        return fields
    
    def load_data(self,db_name,table_name, fields, data):
        cursor = self.cursor
        try:
            cursor.execute(f"USE `{db_name}`;")
            # 根据fields生成建表语句
            create_table_sql = f"CREATE TABLE `{table_name}` ("
            for field in fields:
                create_table_sql += f"`{field}` VARCHAR(255),"
            create_table_sql = create_table_sql[:-1] + ');'
            cursor.execute(create_table_sql)
            # 根据data生成插入语句,data是一个二维数组
            insert_sql = f"INSERT INTO `{table_name}` VALUES "
            for row in data:
                insert_sql += '('
                for field in row:
                    insert_sql += f'"{field}",'
                insert_sql = insert_sql[:-1] + '),'
            insert_sql = insert_sql[:-1] + ';'
            cursor.execute(insert_sql)
            self.conn.commit()
            return True
        except:
            self.conn.rollback()
            return False