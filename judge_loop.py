import time
import dtwzmt
import os
import datetime
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


def loop():
    while True:
        # 从数据库中拿到乐谱地址和用户音频地址
        query = "select id,part_id,audio,ques_type from homework_sightsingingrecord where computer_score is null order by id  limit 1;"
        cursor.execute(query)
        processing_task = cursor.fetchall()

        if len(processing_task) != 1:
            time.sleep(5)
            continue
        # 因为返回的是数组,所以取第一个
        processing_task = processing_task[0]
        # if True:
        try:
            # 进行评分
            task_id = processing_task[0]
            part_id = processing_task[1]
            audio_path = "../solfeggio_django" + processing_task[2].replace('https://musicmuc.chimusic.net', '')
            ques_type = 2 if processing_task[3] == 2 else 1

            sub_query = "select xml_path from question_bank_sightsingingquestion where part_id=%s;" % part_id
            cursor.execute(sub_query)
            xml_path = 'https://musicmuc.chimusic.net' + cursor.fetchall()[0][0]

            txt_save_dir = "melodyExtraction_JDC / results"

            j = dtwzmt.Judge(audio_path=audio_path, xml_path=xml_path,
                             txt_save_dir=txt_save_dir, staff_choice=ques_type)
            j.get_xml_list()
            j.get_voice_list()
            try:
                process_path, score = j.export()
            except FileNotFoundError:
                print("出问题了")
                continue
            # 把存储文件路径存入数据库
            sub_sub_query = "update homework_sightsingingrecord srecord set srecord.computer_score=%s  " \
                            "where srecord.id = %s" % score, task_id
            cursor.execute(sub_query)

        except Exception as e:
            print("读取文件成功了，但是报了错导致分析不成功，原因：", e)


if __name__ == "__main__":
    loop()
