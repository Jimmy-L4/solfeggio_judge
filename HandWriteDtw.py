import matplotlib.pyplot as plt
import music21
import numpy as np
import math
import pandas as pd
import os
import copy
import sys
from sys import getsizeof
import time
import psutil
from numpy.linalg import norm
import time
import merge_audio
import librosa
import fastdtw
import matplotlib

from dtw import dtw

# matplotlib.use('TkAgg')


class PartIndexError(Exception):
    def __init__(self,error):
        self.error = error
    def __str__(self,*args,**kwargs):
        return self.error


def distance(a, b): #如果a或b = 1，  dist = 0， 否则（如果距离在三个半音以内，距离正常算， 否则距离正无穷）
    if b == 1:
        return 10 if a != 1 else 0.04
    else:
        ln_ab = np.abs(np.log(a / b))
        delta_key = 12 * ln_ab / np.log(2)
        if delta_key > 3.2:
            return 10000 * (0.5 + 10 * ln_ab)
        elif delta_key > 2.1:
            return 3000 * (0.5 + 10 * ln_ab)
        elif delta_key > 1.1:
            return 2 * (0.5 + 10 * ln_ab)
        else:
            return max(0.05, 10 * ln_ab)



def plot_dist():
    b = 2
    dist = [distance(b * np.exp(0.01 * i * np.log(2) / 12), b) for i in range(500)]
    plt.scatter([i for i in range(500)], dist)
    plt.show()


class HandWriteDtw:
    # 初始化类
    def __init__(self, voice_list, xml_list, note_dict, bpm, quarterLength):
        self.voice_list = voice_list
        self.xml_list = xml_list  # 记录每个采样点的频率，数组长度与note_dict的endpoint一致
        self.or_note_dict = []
        self.or_voice_dict = []
        self.note_dict = note_dict  # start_point': 1864, 'end_point': 1945, 'note_id': '42', 'freq': 369.99442271163434
        self.voice_dict = []  # 记录人声
        self.match_dict = []
        self.wrong = []
        self.mintime = 0.0
        self.bpm = bpm
        self.get_mintime()
        self.quarterLength = quarterLength
        self.distance = 0
        self.match_dict = None
        # plot_dist()

    '''
    :return:
    1. 普通乐谱，允许包含若干空Voice
    2. 具有双声部的乐谱，允许包含若干空Voice
    3. 只有若干chord的乐谱
    4. 只由若干有意义的Voice组成的乐谱
    5. 在voice中有chord
    6. 在双声部中有voice
    '''


    ''' 谱面处理 '''
    '''包括
        1.删除无用的metadata和text
        2.对measure和note设置x坐标和line属性
        3.剔除多余的全休止和没用的Voice
    '''

    def get_mintime(self):
        min_time = min([note['end_point'] - note['start_point'] for note in self.note_dict if note['freq'] > 0]) / 100
        print(min_time)
        # min_offset = 4
        # min_time = 10.0
        # if self.msc_type == 1 or self.msc_type == 2:
        #     for part in self.msc.getElementsByClass("Part"):
        #         for measure in part.getElementsByClass("Measure"):
        #             for note in measure.getElementsByClass("Note"):
        #                 if note.quarterLength!=0 and note.quarterLength < min_offset:
        #                     min_offset = note.quarterLength
        # #print(min_offset)
        # bpm = int(self.xml_path.split("_")[-1].split(".")[0])
        # #print(bpm)
        # min_time = float(min_offset) * 60 / bpm
        return min_time


    @staticmethod
    def ft2ftrange(ft):
        ft = float(ft)
        if ft < 16:
            return 0, 0
        A4_freq = 440
        A4_pos = 57
        pos = math.log2(ft / A4_freq) * 12 + A4_pos
        posHigh = pos + 0.5
        posLow = pos - 0.5
        ftHigh = A4_freq * (2 ** ((posHigh - A4_pos) / 12))
        ftLow = A4_freq * (2 ** ((posLow - A4_pos) / 12))
        return ftLow, ftHigh

    def VOICE_visualize(self):
        plt.cla()
        X = []
        Y = []
        for i in range(len(self.voice_dict)):
            if i%2==0:
                clr = 'yellow'
            else:
                clr = 'blue'
            for j in range(self.voice_dict[i]['start_point'], self.voice_dict[i]['end_point']+1):
                X.append(j)
                Y.append(self.voice_list[j])
            plt.scatter(X, Y, color=clr, alpha=0.3)
            X = []
            Y = []
        plt.show()


    def XML_visualize(self):
        plt.cla()
        X = []
        Y = []
        for i in range(len(self.note_dict)):
            if i%2==0:
                clr = 'yellow'
            else:
                clr = 'blue'
            for j in range(self.note_dict[i]['start_point'], self.note_dict[i]['end_point']):
                X.append(j)
                Y.append(self.xml_list[j])
            plt.scatter(X, Y, color=clr, alpha=0.3)
            X=[]
            Y=[]
        plt.show()


    def NOTE_visualize(self):
        plt.cla()
        a = len(self.xml_list)
        b = len(self.voice_list)
        cc = max(self.xml_list)
        X1=[]
        X2=[]
        Y1=[]
        Y2=[]
        if a < b: #人声长
            c = float(b)/a
            for i in range(len(self.match_dict)):
                if i%2==0:
                    clr = 'yellow'
                else:
                    clr = 'blue'
                start_point = self.note_dict[i]['start_point']
                end_point = self.note_dict[i]['end_point']
                for j in range(start_point, end_point):
                    X1.append(j)
                    Y1.append(self.xml_list[j])
                start_point2 = self.match_dict[i]['start_point']
                end_point2 = self.match_dict[i]['end_point']
                for j in range(start_point2, end_point2):
                    X2.append(int(j/c))
                    Y2.append(self.voice_list[j]+cc)
                x_axis_data1 = []
                y_axis_data1 = []
                x11 = int((start_point+end_point)/2)
                x21 = int((start_point2+end_point2)/2)
                x_axis_data1.append(x11)
                y_axis_data1.append(self.xml_list[x11])
                x_axis_data1.append(int(x21/c))
                y_axis_data1.append(self.match_dict[i]['beat_mistake_avg']+self.note_dict[i]['freq']+cc)

                plt.scatter(X1, Y1, color=clr, alpha=0.3)
                plt.scatter(X2, Y2, color=clr, alpha=0.3)
                plt.plot(x_axis_data1, y_axis_data1, color='red')

                X1 = []
                X2 = []
                Y1 = []
                Y2 = []
        else:
            c = float(a) / b
            for i in range(len(self.match_dict)):
                if i % 2 == 0:
                    clr = 'yellow'
                else:
                    clr = 'blue'
                start_point = self.note_dict[i]['start_point']
                end_point = self.note_dict[i]['end_point']
                for j in range(start_point, end_point):
                    X1.append(int(j / c))
                    Y1.append(self.xml_list[j])
                start_point2 = self.match_dict[i]['start_point']
                end_point2 = self.match_dict[i]['end_point']
                for j in range(start_point2, end_point2):
                    X2.append(j)
                    Y2.append(self.voice_list[j] + cc)
                x_axis_data1 = []
                y_axis_data1 = []
                x11 = int((start_point + end_point) / 2)
                x21 = int((start_point2 + end_point2) / 2)
                x_axis_data1.append(int(x11/c))
                y_axis_data1.append(self.xml_list[x11])
                x_axis_data1.append(x21)
                y_axis_data1.append(self.match_dict[i]['beat_mistake_avg'] + self.note_dict[i]['freq'] + cc)

                plt.scatter(X1, Y1, color=clr, alpha=0.3)
                plt.scatter(X2, Y2, color=clr, alpha=0.3)
                plt.plot(x_axis_data1, y_axis_data1, color='red')

                X1 = []
                X2 = []
                Y1 = []
                Y2 = []

        plt.show()

    def NOTE_visualize_2(self):
        A = music21.note.Note(69)
        voice_now = 0
        voice_list = self.voice_list
        for s, v in enumerate(voice_list):
            if v > 30:
                voice_list = voice_list[s:]
                break
        # print(self.match_dict[36]['match_list'])
        print(self.match_dict[0]['start_point'])
        print(self.match_dict[0]['end_point'])
        print(len(self.match_dict[0]['match_list']))
        # print(self.match_dict[1]['match_list'])
        for i, d in enumerate(self.match_dict):
            plt.clf()
            points = len(d["match_list"])
            plt.plot([i for i in range(max(0, voice_now - 300), min(voice_now + points + 300, len(voice_list)))],
                     [i - 0.2 for i in voice_list[max(0, voice_now - 300): voice_now + points + 300]])
            plt.plot([i for i in range(voice_now, voice_now + points)], [i - 0.2 for i in voice_list[voice_now: voice_now + points]], c='r')
            plt.axhline(y=self.note_dict[i]['freq'], c='b')
            if self.note_dict[i]['freq'] > 30:
                delta_key = round(np.log(self.note_dict[i]['freq'] / A.pitch.frequency) / np.log((2 ** (1 / 12))))
                label = music21.note.Note(A.pitch.midi + delta_key).name
            else:
                label = "Rest"
            plt.title('note:%s, pitch:%s, points:%s(o)->%s(+%s=%s)'
                % (i + 1, label, "?", points, voice_now, voice_now + points))
            plt.annotate(label, (voice_now, max(300, self.note_dict[i]['freq'] - 100)), fontsize=20, color='blue')
            voice_now += points + 1
            plt.show()

    def Justdtw(self):
        xml_list = self.xml_list.copy()
        voice_list = self.voice_list.copy()
        bpm = self.bpm
        quarterLength = self.quarterLength.copy()
        # quarterLength = [1, 3, 5, 7, 9, 12]
        # quarterLength = [quarterLength[-1] - (quarterLength[len(quarterLength) - i - 2] if i < len(quarterLength) - 1 else 0)for i in range(len(quarterLength))]
        # print(quarterLength)

        match_xmltovoice = []
        # seconds是xml中每个音符的应有秒数
        seconds = [0] + [int(i / bpm * 6000) for i in quarterLength]
        # voice只需首尾去0
        start = end = 0
        for i in range(len(voice_list)):
            if voice_list[i] > 0:
                start = i
                break
        for i in range(len(voice_list)):
            if voice_list[len(voice_list) - 1 - i] > 0:
                end = len(voice_list) - i
                break
        voice_list = voice_list[start: end]
        # 对休止少于短于32分音符（int(15/bpm * 100)）的点的都进行插值处理 效果不好暂时不用
        # def conut_continuous_zero(voice_list, index):
        #     num = 0
        #     for i in range(1, 500):
        #         if index - i < 0:
        #             break
        #         if voice_list[index - i] == 0:
        #             num += 1
        #         else:
        #             break
        #     for i in range(1, 500):
        #         if index + i >= len(voice_list):
        #             break
        #         if voice_list[index + i] == 0:
        #             num += 1
        #         else:
        #             break
        #     return num + 1
        # i = 0
        # while i < len(voice_list):
        #     if voice_list[i] == 0:
        #         zeros = conut_continuous_zero(voice_list, i)
        #         print(f"{i}处有{zeros}个连续的0,正常{int(15/bpm * 100)}")
        #         if zeros < int(15/bpm * 100) + 3:
        #             diff = voice_list[i + zeros] - voice_list[i - 1]
        #             for k in range(zeros):
        #                 # voice_list[i + k] = diff * (k + 1) / (zeros + 1) + voice_list[i + zeros]
        #                 print(i + k, voice_list[i + k], voice_list[i - 1: i + zeros + 1])
        #                 voice_list[i + k] = diff * (k + 1) / (zeros + 1) + voice_list[i - 1]
        #             print(voice_list[i - 1: i + zeros + 1])
        #         i += zeros
        #     else:
        #         i += 1


        # 对 0 都 + 1 处理防止/0报错
        voice_list = [i if i > 0 else 1 for i in voice_list]
        xml_list = [i if i > 0 else 1 for i in xml_list]
        speed_ratio = len(voice_list) / len(xml_list)
        print(len(voice_list), len(xml_list), speed_ratio)
        # 现在已经匹配了voice_now个音频点
        voice_now = 0

        # 一次dtw匹配numOfNote个音符
        numOfNote = 6

        # print(len(xml_list), quarterLength[-1] / bpm * 6000)
        # 在哪些点处切割
        cut_point = []
        # 绘图用 可以删##############

        A = music21.note.Note(69)
        # ##########################
        for i, v in enumerate(seconds):
            # print(i)
            if i >= len(seconds) - numOfNote:
                numOfNote -= 1
                if numOfNote == 1: break
            if True:
                # print("乐谱范围", v, seconds[i + numOfNote])
                # 切八个音符长的片进行匹配
                xml_slice = xml_list[v: seconds[i + numOfNote]]
                # print("音频搜索范围", voice_now, voice_now + int((seconds[i + numOfNote] - v) * 1.75))
                voice_slice = voice_list[voice_now: voice_now + int((seconds[i + numOfNote] - v) * 1 * speed_ratio)]
                # print(len(xml_slice), len(voice_slice))
                # d, path = fastdtw.fastdtw(xml_slice, voice_slice,
                #                           dist=lambda a, b: distance(a, b))
                dtw_time = 0
                while True:
                    dtw_time += 1
                    d, cost_, acc_cost_, path = dtw(xml_slice, voice_slice,
                                  dist=lambda a, b: distance(a, b))
                    path = list(zip(path[0], path[1]))
                    now_xml_len = seconds[i + 1] - seconds[i]
                    # 确定一个音符匹配points个采样点
                    if path[now_xml_len][0] == now_xml_len:
                        points = path[now_xml_len][1]
                        # print(path)
                    else:
                        points = 0
                        for s in range(now_xml_len + 1, seconds[-1]):
                            if path[s][0] == now_xml_len:
                                points = path[s][1]
                                break
                        if points == 0:
                            raise Exception("这个音符对应了0个点，出bug了")
                    # print(voice_now, voice_now + points, points)
                    points = int(points)
                    this_freq = np.mean([i for i in voice_list[voice_now: voice_now + points] if i > 20])
                    delta_key = 12 * np.abs(np.log(this_freq / xml_list[seconds[i] + 1])) / np.log(2)
                    if abs(delta_key) < 1 or dtw_time > 0:
                        self.distance += d / max(len(xml_slice), len(voice_slice))
                        break
                    else:
                        voice_slice = [i * 2 ** (delta_key / 12) for i in voice_slice]

                # 如果连着若干个是一样的音符则尽量按比例分配
                # iter_seconds = i
                # while iter_seconds < len(seconds) - 2:
                #     if -10 < xml_list[seconds[i]+1] - xml_list[seconds[iter_seconds + 1]+1] < 10:
                #         iter_seconds += 1
                #     else:
                #         break
                # 从第i个音符到第iter_seconds个音符（包括）在xml上是同频率的音
                # 第i个音符的标准时间长度是seconds[i + 1] - seconds[i]
                # 你把从i到iter_seconds这些音符的voice视唱加起来 按xml标准视唱的比例分配一下， 给points赋值它应有的点数就行

                # if iter_seconds > i:
                #     total_long = seconds[iter_seconds + 1] - seconds[i]  # xml总点数
                #     goal_total_point = seconds[iter_seconds + 1]
                #     for index in range(goal_total_point, len(path)):
                #         # print(path[index][0], seconds[iter_seconds + 1])
                #         if path[index][0] > seconds[iter_seconds + 1]:
                #             goal_total_point = index - 1
                #             break
                #     voice_long = path[goal_total_point][1] - path[seconds[i]][1]  # voice总点数
                #     new = int((seconds[i + 1] - seconds[i]) / total_long * voice_long) # 第一个音符按照比例分配的长度
                #     points = new
                #     print(path)
                #     print(path[seconds[i]][1], seconds[i], i, points, (seconds[i + 1] - seconds[i]), total_long, voice_long)
                    #                     # print(1/0)


                # 过长/过短调整
                if (points > 1.8 * now_xml_len * speed_ratio or points * 1.8 < now_xml_len * speed_ratio) and xml_list[seconds[i]+1] > 30:
                    min_cost = 999999
                    min_points = 5
                    for h in range(max(6, int(now_xml_len * speed_ratio / 2)), int(now_xml_len * speed_ratio * 2)):
                        # 时长损失
                        real_voice_len = now_xml_len * speed_ratio
                        time_cost = abs(h - real_voice_len) - sum([1 if i < 30 else 0 for i in voice_list[voice_now: voice_now + h]])
                        time_cost = max(0.0,
                                        time_cost - 0.25 * real_voice_len + 0.075 * real_voice_len
                                        if time_cost > 0.25 * real_voice_len
                                        else 0.3 * time_cost)

                        # 音高损失
                        this_voice_pitch = [s for s in voice_list[voice_now: voice_now + h] if s > 20]
                        # print("this", this_voice_pitch, len(this_voice_pitch))
                        this_voice_pitch = np.mean(this_voice_pitch)
                        pitch_cost = np.abs(np.log(this_voice_pitch / xml_list[seconds[i]+1]))
                        pitch_cost = 0.02 if 2 ** (1/36) > this_voice_pitch / xml_list[seconds[i]+1] > 1 / (2 ** (1/36)) else pitch_cost
                        # 方差损失
                        var_cost = np.var(np.log([s for s in voice_list[voice_now: voice_now + h] if s > 20])) * 10

                        # 下一个音符的音高损失
                        next_xml_pitch = xml_list[seconds[i + 1] + 1]

                        next_voice_pitch = []
                        # print(voice_list[voice_now+h-3:voice_now+h+2])
                        for s in range(voice_now + h, len(voice_list)):
                            if voice_list[s] > 20:
                                next_voice_pitch .append(voice_list[s])
                                # if len(next_voice_pitch) > seconds[i+2 if i + 2 < len(seconds) else -1] - seconds[i+1]:
                                if len(next_voice_pitch) > 10:
                                    break
                        # print("next", next_voice_pitch, len(next_voice_pitch))
                        next_voice_pitch = np.mean(next_voice_pitch)
                        pitch_cost_next = np.abs(np.log(next_voice_pitch / next_xml_pitch))
                        # print(next_voice_pitch, next_xml_pitch, this_voice_pitch, xml_list[seconds[i]+1])
                        if 2 ** (1 / 36) > (next_voice_pitch / next_xml_pitch) > 1 / (2 ** (1 / 36)):
                            pitch_cost_next = 0.02
                        # pitch_cost_next = 0.05 if 2 ** (1/36) > (next_voice_pitch / next_xml_pitch) > 1 / (2 ** (1/36)) else pitch_cost_next

                        cost = 0.0006 * time_cost ** 2 + pitch_cost * 20 + var_cost * 10 + pitch_cost_next * 20
                        # print(h, cost, 0.0006 * time_cost ** 2, pitch_cost * 20, var_cost, pitch_cost_next * 20)

                        if cost < min_cost:
                            # print(h, abs(h - now_xml_len * speed_ratio), now_xml_len * speed_ratio, time_cost, pitch_cost, var_cost, 0.3 * time_cost ** 2, pitch_cost * 10, var_cost)
                            min_cost = cost
                            min_points = h
                            # print(this_voice_pitch, xml_list[seconds[i]+1], next_voice_pitch, next_xml_pitch)

                    # print(time_cost, pitch_cost, var_cost, cost)
                    print("改了:", min_points, min_cost, f'time:{0.0006 * time_cost ** 2}, pitch:{pitch_cost * 20}, var:{var_cost * 10}, pitch_next:{pitch_cost_next * 20}')
                    points = min_points
                    # print(voice_list[voice_now + points - 10: voice_now + points], voice_list[voice_now + points: voice_now + points + 10])
                #
                #     # print(points)
                #     # print(path)
                # if xml_list[seconds[i]+1] > 30 and voice_list[voice_now + points - 1] > 30:
                #     z_count = False
                #     last_check = 1
                #     for p in range(1, int(2 * points / 3)):
                #         if z_count and voice_list[voice_now + points - 1 - p] > 30:
                #             last_check = p
                #             z_count = False
                #         if not z_count and voice_list[voice_now + points - 1 - p] < 30:
                #             z_count = True
                #     points -= last_check - 1

                # 画图
                # plt.clf()
                #
                # plt.plot([i for i in range(max(0, voice_now - 300), min(voice_now + points + 300, len(voice_list)))],
                #          [i - 0.2 for i in voice_list[max(0, voice_now - 300): voice_now + points + 300]])
                # plt.plot([i for i in range(voice_now, voice_now + points)], [i - 0.2 for i in voice_list[voice_now: voice_now + points]], c='r')
                # plt.axhline(y=xml_list[seconds[i]+1], c='b')
                # if xml_list[seconds[i]+1] > 30:
                #     delta_key = round(np.log(xml_list[seconds[i]+1] / A.pitch.frequency) / np.log((2 ** (1 / 12))))
                #     label = music21.note.Note(A.pitch.midi + delta_key).name
                # else:
                #     label = "Rest"
                # plt.title('note:%s, pitch:%s, points:%s(o)->%s(+%s=%s)'
                #     % (i + 1, label, now_xml_len, points, voice_now, voice_now + points))
                # plt.annotate(label, (voice_now, max(300, xml_list[seconds[i]+1] - 100)), fontsize=20, color='blue')
                # if i > -1:
                #     plt.show()
                # if i in [35]:
                #     print(len(voice_list[voice_now: voice_now + points]), voice_list[voice_now: voice_now + points])
                voice_now += points
                cut_point.append(voice_now)
        print(self.distance, len(seconds) - numOfNote, self.distance / (len(seconds) - numOfNote))
        self.distance = self.distance / (len(seconds) - numOfNote)
        return cut_point

    @staticmethod
    def examine_note(be_zero, a,cut_points,voice_list,ratio_stand):
        def minus(x,ratio_stand):
            result = -1
            dis = 1
            for i in range(len(ratio_stand)):
                if abs(ratio_stand[i]-x)<dis:
                    result = i
                    dis = abs(ratio_stand[i]-x)
            return result

        def get_cut(x,y,new_cut2):
            #print(x,y,ratio_stand)
            c = []
            total_long = 0
            cut_long = 0
            if x>0 and y<len(ratio_stand)-1:
                total_long = ratio_stand[y+1] - ratio_stand[x-1]
                cut_long = cut_points[a[0]+y+1] - cut_points[a[0]+x-1]
            elif x>0 :
                total_long = 1.0 - ratio_stand[x-1]
                cut_long = cut_points[a[1]] - cut_points[a[0]+x-1]
            elif x ==0 and y == len(ratio_stand)-1:
                total_long = 1.0
                cut_long = cut_points[a[1]] - cut_points[a[0]-1] if a[0]>0 else cut_points[a[1]]
            elif x==0 and y<len(ratio_stand)-1:
                total_long = ratio_stand[y+1]
                cut_long = cut_points[a[0]+y+1]

            for i in range(x,y+1):
                # note_long = ratio_stand[x] - ratio_stand[x-1] if x>1 else ratio_stand[x]
                c.append(ratio_stand[i]/total_long * cut_long)

            # for i in range(1,len(c)):
            #     c[i] += c[i-1]

            if a[x] == 0:
                for i in range(len(c)):
                    new_cut2.append(int(c[i]))
            else:
                for i in range(len(c)):
                    new_cut2.append(int(c[i])+cut_points[a[x]-1])


        intervals = a[1]-a[0]
        num = 0
        new_cut = []
        be = 0
        en = 0
        for i in range(a[0],a[1]):  ##如果休止点太靠近始末就不算
            if voice_list[be_zero+i]>0 and voice_list[be_zero+i+1]==0:
                if i - a[0] < 10:
                    continue
                else:
                    be = i
            if voice_list[be_zero+i] == 0 and voice_list[be_zero+i+1]>0:
                if i - a[1] < 10:
                    continue
                else:
                    en = i
            if en > 0:
                if be > 0:
                    num += 1
                    new_cut.append([be,en])
                else:
                    en = 0

        if a[0] > 1:
            voice_long = cut_points[a[1]] - cut_points[a[0]-1]
        else:
            voice_long = cut_points[a[1]]
        if num == intervals:
            return new_cut[:,0]
        elif num<intervals:
            new_cut2 = []
            cut = []
            for i in range(len(new_cut)):
                x = float(new_cut[i][0])/voice_long
                #ratio_cut.append(x)#休止符比例
                cut.append(minus(x,ratio_stand))#已知的切割点,标准音符的下标，0，1，2
            cc = 0#已经确定的分割点
            temp = 0
            while(cc < intervals):
                if temp >= len(cut)-1:
                    get_cut(cc, intervals-1,new_cut2)
                    cc = intervals
                else:
                    if cut[temp]==cc:
                        new_cut2.append(new_cut[cc][0])
                        cc+=1
                        temp+=1
                    elif cut[temp] > cc:
                        get_cut(cc, cut[temp]-1 ,new_cut2)#cc结尾不知道，cut[temp]结尾知道，cut[temp]-1结尾不知道
                        cc = cut[temp]
                        temp += 1
                    else:
                        print("wrong")
            return new_cut2
        else:
            return []




    def DTW(self):
        voice_list = self.voice_list
        cut_points = self.Justdtw()
        print(cut_points)
        ###开始检查相同的连续音符
        cut_points2 = cut_points.copy()
        m = 0
        examine = []
        first = 0
        last = 0
        while m < len(self.note_dict) - 1:
            # print(m, len(self.note_dict), self.note_dict[m+1])
            if self.note_dict[m]['freq'] == self.note_dict[m+1]['freq']:
                first = m
                m = m + 1
                while m < len(self.note_dict) - 1:
                    if self.note_dict[m]['freq']==self.note_dict[m+1]['freq']:
                        m += 1
                    else:
                        last = m
                        # if last - first <=1:
                        #     break
                        # else:
                        examine.append([first,last])
                        break
            else:
                m += 1
        print(examine)

        ##统计voice_list开头的0的个数
        be_zero = 0
        for i in range(len(self.voice_list)):
            if self.voice_list[i] == 0:
                be_zero += 1
            else:
                break
        #print(be_zero)


        for i in range(len(examine)):
            note_long = float(self.note_dict[examine[i][1]]['end_point'] - self.note_dict[examine[i][0]]['start_point'])
            ratio_stand = []
            for j in range(examine[i][0],examine[i][1]+1):
                ratio_stand.append((self.note_dict[j]['end_point']-self.note_dict[j]['start_point'])/note_long)
            for j in range(1,len(ratio_stand)):
                ratio_stand[j] += ratio_stand[j-1]
            ratio_stand.pop()
            #print(ratio_stand)
            new_cut = self.examine_note(be_zero, examine[i],cut_points,self.voice_list,ratio_stand)
            if(len(new_cut)>0):
                for j in range(len(new_cut)):
                    cut_points2[j+examine[i][0]] = new_cut[j]

        cut_points = cut_points2
        print(cut_points)

        ##检查结束返回结果

        voice_first = 0
        voice_last = 0
        if first > 0:
            voice_first = cut_points[first - 1] + 1
        voice_last = cut_points[last]
        xml_long = -self.note_dict[first]['start_point'] + self.note_dict[last]['end_point']
        voice_long = voice_last - voice_first
        for j in range(first, last + 1):
            if j == 0:
                cut_points2[j] = int(
                    (self.note_dict[j]['end_point'] - self.note_dict[j]['start_point']) / xml_long * voice_long)
            else:
                cut_points2[j] = cut_points[j - 1] + 1 + int(
                    (self.note_dict[j]['end_point'] - self.note_dict[j]['start_point']) / xml_long * voice_long)

        # cut_2 = self.BiJustdtw()
        # start = end = 0
        # for i in range(len(voice_list)):
        #     if voice_list[i] > 0:
        #         start = i
        #         break
        # for i in range(len(voice_list)):
        #     if voice_list[len(voice_list) - 1 - i] > 0:
        #         end = len(voice_list) - i
        #         break
        # plot_list = voice_list[start: end]
        # plt.clf()
        # for i in cut_points:
        #     plt.axvline(x=i, c='r', alpha=0.3)
        # for i in cut_2:
        #     plt.axvline(x=len(plot_list) - i, c='g', alpha=0.3)
        # plt.show()

        len_note = len(self.note_dict)


        match_dict = []
        note_id = 0
        note_xml_st = 0
        note_xml_ed = 0
        just = 0


        for i in range(len(cut_points)):
            note_xml_st = just
            note_xml_ed = cut_points[i] + be_zero -1
            just = note_xml_ed +1
            nozero = 1
            for j in range(note_xml_st, note_xml_ed + 1):
                if voice_list[j] > 0:
                    nozero += 1
            # print(note_xml_st,note_xml_ed)
            # 第一个单独处理
            if i == 0:
                match_list = self.voice_list[note_xml_st: note_xml_ed]
                if len(match_list) > 5:
                    for i, v in enumerate(match_list):
                        if v > 20:
                            match_list = match_list[i:]
                            break
                match_dict.append(
                    {"note_id": str(note_id + 1), "match_list": match_list,
                     "beat_mistake_fir": note_xml_st - self.note_dict[note_id]['start_point'],
                     "beat_mistake_end": note_xml_ed - self.note_dict[note_id]['end_point'] + 1,
                     "beat_mistake_avg": np.sum(voice_list[int(note_xml_st): int(note_xml_ed) + 1]) / nozero - self.note_dict[note_id]['freq'],
                     "start_point": note_xml_st,
                     "end_point": note_xml_ed})
            else:
                match_dict.append(
                    {"note_id": str(note_id + 1), "match_list": self.voice_list[note_xml_st: note_xml_ed],
                     "beat_mistake_fir": note_xml_st - self.note_dict[note_id]['start_point'],
                     "beat_mistake_end": note_xml_ed - self.note_dict[note_id]['end_point'] + 1,
                     "beat_mistake_avg": np.sum(voice_list[int(note_xml_st): int(note_xml_ed) + 1]) / nozero - self.note_dict[note_id]['freq'],
                     "start_point": note_xml_st,
                     "end_point": note_xml_ed})
            #print(note_xml_st,note_xml_ed)
            note_id += 1
        nozero = 1
        for j in range(just, len(self.voice_list)):
            if voice_list[j] > 0:
                nozero += 1
        match_list = voice_list[int(just): int(len(self.voice_list))]
        if len(match_list) > 5:
            for i, v in enumerate(match_list[::-1]):
                if v > 20:
                    match_list = match_list[i:]
                    match_list = match_list[::-1]
                    break
        match_dict.append(
            {"note_id": str(note_id + 1), "match_list": match_list,
             "beat_mistake_fir": len(self.voice_list) - 1 - self.note_dict[note_id]['start_point'],
             "beat_mistake_end": len(self.voice_list) - 1 - self.note_dict[note_id]['end_point'] + 1,
             "beat_mistake_avg": np.sum(voice_list[int(just): int(len(self.voice_list))]) / nozero -
                                 self.note_dict[note_id]['freq'],
             "start_point": min(just, len(self.voice_list) - 1),
             "end_point": len(self.voice_list) - 1})
        #print(voice_list[0:97])
        # print(len(match_dict[35]["match_list"]), match_dict[35]["match_list"])
        self.match_dict = match_dict
        # self.NOTE_visualize_2()
        return match_dict








    #对声音序列处理
