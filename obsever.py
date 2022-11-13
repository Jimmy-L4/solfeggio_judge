# 文件夹监控器
# 没有采用,还是使用了数据库的方式
# 2022/11/1
import re
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
import dtwzmt

import pymysql

# 建立一个MySQL连接
conn = pymysql.connect(
    host='10.112.59.2',
    user='root',
    passwd='bupt2021',
    db='solfeggio_django',
    port=3306,
    charset='utf8mb4'
)
cursor = conn.cursor()


# 重写看门狗🐶的create方法
class extractor(LoggingEventHandler):

    def on_created(self, event):
        super(LoggingEventHandler, self).on_created(event)
        what = 'directory' if event.is_directory else 'file'
        logging.info("Created %s: %s", what, event.src_path)
        name_ext = event.src_path.split('.')
        if name_ext[-1] == 'wav':
            # example:
            # ../solfeggio_django/media/audio/002000000000000020101/002000000000000020101-22.wav
            file_info = re.split("[./-]", event.src_path)
            user_id = file_info[-2]
            part_id = file_info[-3]
            logging.info("wav文件, 机器评分中...")
            print(user_id, part_id)
            time.sleep(15)

    def launch_judeg(self, part_id):
        sub_query = "select xml_path from question_bank_sightsingingquestion where part_id=%s;" % part_id
        cursor.execute(sub_query)
        xml_path = cursor.fetchall()[0][0]

        txt_save_dir = "melodyExtraction_JDC / results"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    # path = sys.argv[1] if len(sys.argv) > 1 else '.'
    path = '../solfeggio_django/media/audio/'
    # 生成事件处理器对象
    event_handler = extractor()

    # 生成监控器对象
    observer = Observer()
    # 注册事件处理器
    observer.schedule(event_handler, path, recursive=True)
    # 监控器启动——创建线程
    observer.start()

    # 以下代码是为了保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    # 主线程任务结束之后，进入阻塞状态，一直等待其他的子线程执行结束之后，主线程再终止
    observer.join()
