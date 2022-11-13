from pymysql import connect

def run_cursor(sql, par=None):
    cc = SC()
    cs = cc.cursor
    # cs.execute('use ' + database_name)
    try:
        if not par:
            # sql = sql.replace('\'%s\'', '%s').replace('\"%s\"', '%s')
            cs.execute(sql)
        else:
            sql = sql.replace('\'%s\'', '%s').replace('\"%s\"', '%s')
            cs.execute(sql, par)
        cc.conn.commit()
    except Exception as e:
        print(str(e))
        cc.conn.rollback()
        if par:
            print(1, "出错的sql为%s" % (sql % par))

        else:
            print(2, "出错的sql为%s" % sql)
        return "Error,sql=%s" % sql
    return cs.fetchall()
