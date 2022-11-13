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
import pickle
import ctypes

from dtw import dtw
from HandWriteDtw import HandWriteDtw


# import dtw


class PartIndexError(Exception):
    def __init__(self, error):
        self.error = error

    def __str__(self, *args, **kwargs):
        return self.error


class Judge:
    # 初始化类
    def __init__(self, audio_path: str, xml_path: str, txt_save_dir: str, staff_choice: int):
        # 绝对化音频地址
        self.audio_path = audio_path
        if not os.path.isabs(audio_path):
            self.audio_path = os.path.join(os.path.abspath("."), audio_path)
        merge_audio.match_target_amplitude(audio_path)
        # 绝对化xml地址
        self.xml_path = xml_path
        if not os.path.isabs(xml_path):
            self.xml_path = os.path.join(os.path.abspath("."), xml_path)
        # 绝对化文件保存路径
        self.txt_save_dir = txt_save_dir
        if not os.path.isabs(txt_save_dir):
            self.txt_save_dir = os.path.join(os.path.abspath("."), txt_save_dir)

        self.msc = music21.converter.parse(self.xml_path)
        self.judge_staff = staff_choice
        self.text = ""
        self.quarterLength = []
        self.distance = 0
        self.note = []
        self.k_measure_ratio = []

        try:
            self.speed = xml_path.split("_")[-1].split(".")[0]
            # TODO 这部分是获取基础节拍的代码（以四分/八分音符为一拍，暂未使用）
            try:
                measure = self.msc.getElementsByClass("Part")[0]
                time_signature = measure.getElementsByClass("TimeSignature")[0]
                print(time_signature.numerator, time_signature.denominator, time_signature.denominator)
            except IndexError:
                pass
            if not self.speed.isdigit():
                self.speed, self.Italian = self.speed.split()
                # TODO 目前有些速度给的是以8分音符为一拍的bpm， 但没给TimeSignature， 只能先单独判断
                if self.xml_path.endswith("隔河相思_1_32_116 Andantino.xml"):
                    self.speed = 116
            self.speed = int(self.speed)
        except IndexError:
            self.speed = None
        self.msc_type = self.check_type_of_msc()
        self.score = 0
        self.total_line = 0
        self.voice_list = []
        self.xml_list = []  # 记录每个采样点的频率，数组长度与note_dict的endpoint一致
        self.note_dict = []  # start_point': 1864, 'end_point': 1945, 'note_id': '42', 'freq': 369.99442271163434
        self.note_dict2 = []  # 记录人声
        self.result1 = []
        self.result2 = []
        self.order = []
        self.pitch_score = 0
        self.start_point = 0
        self.end_point = -1
        self.beats = 0
        self.beat_type = 0
        self.start_mute = 0  # ms
        self.single_note_judge = []
        if self.judge_staff > len(self.msc.parts):
            raise PartIndexError("该乐谱没有二声部")

    '''
    :return:
    1. 普通乐谱，允许包含若干空Voice
    2. 具有双声部的乐谱，允许包含若干空Voice
    3. 只有若干chord的乐谱
    4. 只由若干有意义的Voice组成的乐谱
    5. 在voice中有chord
    6. 在双声部中有voice
    '''

    def check_type_of_msc(self):
        msc_type = len(self.msc.getElementsByClass("Part"))
        has_vaild_voice = 0
        has_chord = 0
        has_staff2 = 0
        if msc_type == 2:
            has_staff2 = 1
        for part in self.msc.getElementsByClass("Part"):
            for measure in part.getElementsByClass("Measure"):

                for v in measure.getElementsByClass("Voice"):
                    for n in v.getElementsByClass(["Note", "Chord", "Rest"]):

                        # self.note.append(id(n))
                        self.note.append(n)
                        if n.isChord:
                            # print("完蛋，为什么voice里还夹chord， 我处理不了")
                            self.msc_type = 5
                            has_chord = 1
                            has_vaild_voice = 1
                        elif n.isRest:
                            continue
                        else:
                            has_vaild_voice = 1
                            # print(v, "这是voice??")
                            if msc_type == 2:
                                # print("就离谱，双声部里还有带音符的Voice")
                                self.msc_type = 6
                                if "政策落实人心爽" in self.xml_path:
                                    self.msc_type = 2
                            else:
                                msc_type = 4
                for note in measure.getElementsByClass(["Note", "Chord", "Rest"]):
                    # self.note.append(id(note))
                    self.note.append(note)
                    # print([i for i in part.flat])
                    if note.isChord:
                        has_chord = 1
                        if msc_type == 4:
                            # print("完蛋，为什么有voice还有chord")
                            pass
                        elif msc_type == 2:
                            # print("就离谱，双声部里还有chord")
                            pass
                        else:
                            if len(measure.getElementsByClass(["Note"])) == 0:
                                msc_type = 3
                    elif note.isRest:
                        continue
                    else:
                        continue
        if has_vaild_voice + has_chord + has_staff2 > 1:
            # print(Warning("这个乐谱具有多个特殊属性，voice=%s, chord=%s, staff2=%s" % (has_vaild_voice, has_chord, has_staff2)))
            pass
        return msc_type

    ''' 谱面处理 '''
    '''包括
        1.删除无用的metadata和text
        2.对measure和note设置x坐标和line属性
        3.剔除多余的全休止和没用的Voice
    '''

    def msc_process(self):
        self.get_measure_order()
        self.flatten_msc()

        """1.删除无用的metadata和text"""
        # 清空metadata, 可能会有问题
        self.msc.metadata = music21.metadata.Metadata()
        # 清空text
        len_msc = len(self.msc)
        for ind in range(len_msc):
            i = (len_msc - 1) - ind
            if type(self.msc[i]) == music21.text.TextBox:
                self.msc.pop(i)

        """2.对measure和note设置x坐标和line属性"""
        # 获取pagewidth， 默认1200 - lmargin - rmargin - system_margin
        gl = music21.layout.divideByPages(self.msc)
        all = gl.getAllMeasurePositionsInDocument()
        margins = gl.getMarginsAndSizeForPageId(0)
        system_margin = 18
        page_width = margins.width - margins.left - margins.right - system_margin

        for part in self.msc.getElementsByClass("Part"):
            ''' 这个部分用来获取layout，是获取x的准备工作'''
            ith_measure_layout_info = -1
            now_top = 0
            now_line = 0

            # 出图的总pixel
            total_pixel = 1

            # TODO #############获取乐谱速度#######################
            # 目前的算法只支持获取乐谱中出现的第一个速度, 所以只有range(1)
            # 以后需要补充变速时的情况
            # music21.tempo类文档 ↓
            # http://web.mit.edu/music21/doc/moduleReference/moduleTempo.html#music21.tempo.MetronomeMark
            if not self.speed:
                metronome_tuples = part.metronomeMarkBoundaries()
                # for i in range(len(metronome_tuples)):
                for i in range(1):
                    try:
                        metronome_tuple = metronome_tuples[i]
                        start_offset, end_offset, metronome_mark = metronome_tuple
                        self.speed = metronome_mark.number
                    except IndexError:
                        self.speed = 120
            # TODO ###############################################
            for measure in part.getElementsByClass("Measure"):
                '''获取measure.line'''
                ith_measure_layout_info += 1
                # print(all[ith_measure_layout_info][0]['top'], now_top, measure.number)
                if all[ith_measure_layout_info][0]['top'] > now_top:
                    now_top = all[ith_measure_layout_info][0]['top']
                    # TODO 换行
                    now_line += 1
                measure.line = now_line
                ''''''
                """3.删除voice"""
                if self.msc_type < 4:
                    for v in measure.getElementsByClass("Voice"):
                        for n in v.getElementsByClass(["Note", "Chord", "Rest"]):
                            pass
                        measure.pop(measure.index(v))
                """"""
                last_x = 0
                for note in measure.getElementsByClass(["Note", "Chord", "Rest"]):
                    if note.isChord:
                        absoluteX = note.notes[0].style.absoluteX
                        try:
                            # TODO 已知问题：harmony的音符获取不到absolutex###########
                            print(Warning(
                                "这里有harmony需要单独处理,乐谱类型%s, 乐谱地址%s" % (self.msc_type, self.xml_path)))
                            absoluteX = 1
                            # #################################################33###
                        except IndexError:
                            pass
                    elif note.isRest:
                        """3.删除小节休止"""
                        if note.fullName == "Whole Rest":
                            measure.pop(measure.index(note))
                            continue
                        absoluteX = int(last_x * page_width) + 45
                    else:
                        vaild_note = note
                        while not vaild_note.style.absoluteX:
                            vaild_note = vaild_note.next("Note")
                            # print(vaild_note)
                        absoluteX = vaild_note.style.absoluteX
                    '''获取note.x, note.line'''
                    note.line = measure.line
                    note.x = (all[ith_measure_layout_info][0]['left'] - margins.left - system_margin +
                              absoluteX) * (total_pixel / page_width)
                    if note.isChord:
                        for n in note.notes:
                            n.x = note.x
                    last_x = note.x
                    ''''''

            ''''''
            self.total_line = now_line

    def get_measure_order(self):
        # 目前通过房子线来确定order
        if "7_课次8_1_34_90" in self.xml_path:
            # 这首是1.2同房子需要单独处理
            self.order = [1, 29, 1, 27, 30, 34]
            # print(self.audio_path)
            return
        repeat_measure = []
        home = []
        if self.msc_type in [1, 2, 4]:
            # 先找一遍有没有反复小节、反复记号
            first_part = self.msc.getElementsByClass("Part")[0]
            try:
                # 如果以后需要改的话可以通过RepeatBracket来确定房子范围
                for RB in first_part.getElementsByClass("RepeatBracket"):
                    pass
            except:
                pass
            repeat_start = 0
            repeat_end = 0
            total_measure = len(first_part.getElementsByClass("Measure"))
            for measure in first_part.getElementsByClass("Measure"):
                for barline in measure.getElementsByClass("Barline"):
                    if barline.type == "regular":
                        # 有regular小节线！近似认为进入房子
                        home.append(measure.number)
                for repeat in measure.getElementsByClass("Repeat"):
                    if repeat.direction == "start":
                        repeat_measure.append(measure.number)
                        repeat_start = measure.number
                    elif repeat.direction == "end":
                        if repeat_start == 0:
                            repeat_measure.append(1)
                        repeat_measure.append(measure.number)
                        repeat_end = measure.number
                    else:
                        print("啥？这个属性不是start也不是end？")
            if len(repeat_measure) % 2 == 1:
                repeat_measure.append(first_part.getElementsByClass("Measure")[-1].number)
            now_home = 0
            if len(repeat_measure):
                order = [1]
            else:
                order = []
            for i in range(len(repeat_measure) // 2):
                start = repeat_measure[i * 2]
                end = repeat_measure[2 * i + 1]
                order.append(end)
                order.append(start)
                if now_home < len(home) and start < home[now_home] <= end:
                    order.append(home[now_home] - 1)
                    now_home += 1
                    order.append(home[now_home])
                    now_home += 1
            order.append(total_measure)
            if repeat_measure:
                self.order = order

    def flatten_msc(self):
        if not self.order:
            return
        else:
            def clear_measure1_layout(measure):
                new_measure = music21.stream.Measure()
                new_measure.layoutWidth = measure.layoutWidth
                for not_layer_ele in measure.elements:
                    if type(not_layer_ele) in [music21.layout.PageLayout,
                                               music21.layout.StaffLayout, music21.clef.TrebleClef,
                                               music21.key.Key, music21.meter.TimeSignature,
                                               music21.expressions.TextExpression, music21.tempo.MetronomeMark]:
                        # ls = layout.divideByPages(lt, fastMeasures=True)
                        # ls = music21.layout.divideByPages(self.msc)
                        # print(ls.getPositionForStaffMeasure(0, measure.number))
                        pass
                    elif type(not_layer_ele) == music21.layout.SystemLayout:
                        for i in range(2, len(self.order), 2):
                            if self.order[i] == 1:
                                return_measure_number = self.order[i - 1]
                                measure_end_x = music21.layout.divideByPages(self.msc). \
                                    getPositionForStaffMeasure(0, return_measure_number)[1][1]
                                if measure_end_x > 550:
                                    not_layer_ele.isNew = True
                                break
                        new_measure.append(not_layer_ele)
                    else:
                        new_measure.append(not_layer_ele)
                return new_measure

            # print(self.order, self.xml_path, self.msc_type)
            for part_index in range(len(self.msc.parts)):
                part = self.msc.parts[0]

                # print(part.quarterLength, part.elements)
                no_layout_measure1 = clear_measure1_layout(part.getElementsByClass("Measure")[0])
                pair_index = 0
                start, end = self.order[pair_index * 2], self.order[pair_index * 2 + 1]
                if len(self.msc.parts) == 1:
                    new_part = music21.stream.Part()
                else:
                    new_part = music21.stream.PartStaff()
                i = 0

                measure_count = 0
                for ele in part.elements:
                    if type(ele) == music21.stream.Measure:
                        if measure_count == 1:
                            break
                        else:
                            measure_count += 1
                    else:
                        new_part.append(copy.deepcopy(ele))

                while i < len(part.elements):
                    ele = part.elements[i]
                    if type(ele) == music21.stream.Measure:
                        if start <= ele.number <= end:
                            if ele.number == 1 and len(new_part.getElementsByClass("Measure")):
                                copy_measure = copy.deepcopy(no_layout_measure1)
                            else:
                                copy_measure = copy.deepcopy(ele)
                            copy_measure.number = len(new_part.getElementsByClass("Measure")) + 1
                            for repeat in copy_measure.getElementsByClass("Repeat"):
                                rp_ind = copy_measure.index(repeat)
                                copy_measure.pop(rp_ind)
                            new_part.append(copy_measure)
                            if ele.number == end:
                                pair_index += 1
                                try:
                                    start, end = self.order[pair_index * 2], self.order[pair_index * 2 + 1]
                                except IndexError:
                                    break
                                i = 0
                                continue
                    i += 1
                # print(new_part.quarterLength, new_part.elements)
                part_ind = self.msc.index(part)
                part_offset = part.offset
                self.msc.pop(part_ind)
                self.msc.insert(part_offset, new_part)
                # self.msc.parts[pair_index] = new_part
                # new_part.show()
                # print(1/0)
                pass

            fp = self.msc.write("musicxml", fp=os.path.join(self.txt_save_dir, "temp.xml"))
            self.msc = music21.converter.parse(fp)
            pass

    # 获取录制音频的频率数组
    def get_voice_list(self):
        # 将mp3转换成以0.01s为单位的频率存储进txt
        def mp3totxt(proj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), r'melodyExtraction_JDC')):
            txt_path = os.path.join(self.txt_save_dir, '%s.txt' % os.path.basename(self.audio_path).replace(".mp3", ""))
            if os.path.exists(txt_path):
                print("txt exists")
                return
            try:
                try:
                    from melodyExtraction_JDC import fix_JDC
                    fix_JDC.main(filepath=self.audio_path, output_dir=self.txt_save_dir, gpu_index=1)
                except ImportError:
                    os.chdir(proj_path)
                    os.system("python melodyExtraction_JDC.py -p %s -o %s" % (self.audio_path, self.txt_save_dir))
                    print("import方式不成功，使用命令行方式运行JDC")
            except Exception:
                raise FileNotFoundError("MP3totxt函数出错。 尝试修改melodyExtraction_JDC工程地址？")

        def read_voice(txt_save_dir=self.txt_save_dir, audio_path=self.audio_path) -> [float]:
            # 从音频文件生成的文本地址中读取数组
            with open(os.path.join(txt_save_dir, '%s.txt' % os.path.basename(audio_path).replace(".mp3", "")),
                      "r") as f:
                res = []
                for line in f.readlines():
                    line = line.strip('\n')
                    str1, str2 = line.split(" ")
                    # print(str2,str1)
                    res.append(float(str2))
                return res

        mp3totxt()
        voice_list = read_voice()
        y, sr = librosa.load(self.audio_path)
        # print(np.max(np.abs(y)) * 0.1)
        len_voice_list = len(voice_list)

        self.voice_list = voice_list
        return voice_list

    # 获取谱例频率数组 xml_list, notes_dict
    def get_xml_list(self):
        self.msc_process()
        number = 1
        xml_list = []
        notes_dict = []

        if self.msc_type == 1 or self.msc_type == 2:
            for part in self.msc.getElementsByClass("Part"):
                if number != self.judge_staff:
                    number += 1
                    continue
                number += 1

                note_count = 1
                now_length = 0

                repeat_note = []
                repeat_dict = []
                repeat_flag = False
                ending = False
                total_quarter_length = part.quarterLength
                total_time = int(((total_quarter_length * 60) / self.speed) * 100)
                # print(total_time)
                xml_list = [0 for _ in range(total_time)]
                last_measure_length = 0
                for measure in part.getElementsByClass("Measure"):
                    measure_msc_length = measure.quarterLength
                    try:
                        time_signature = measure.getElementsByClass("TimeSignature")[0]
                        beats = time_signature.numerator
                        beat_type = time_signature.denominator
                    except IndexError:
                        beats = beat_type = 0
                    if beats != 0 and beat_type != 0:
                        self.beat_type = beat_type
                        self.beats = beats
                        measure_actual_length = beats / beat_type * 4
                        offset_error = measure_msc_length - measure_actual_length
                        measure_length = measure_actual_length
                    else:
                        measure_length = last_measure_length

                    if measure_length == 0:
                        print("measure长度为0，出问题了。")
                    else:
                        last_measure_length = measure_length
                        if measure_msc_length == 0 and measure_msc_length <= 4:
                            now_length += measure_length
                    # print(measure_msc_length, measure_actual_length)
                    # print(measure_msc_length, measure_length)
                    for note in measure.getElementsByClass(["Note", "Chord", "Rest"]):
                        # TODO 这把vaild_note检测删了，试试之后用不用加
                        start_time = int(((now_length * 60) / self.speed) * 100)
                        now_length += note.quarterLength
                        self.quarterLength.append(now_length)
                        end_time = int(((now_length * 60) / self.speed) * 100)
                        note.id = str(note_count)
                        note_count += 1
                        if note.isChord:
                            raise AttributeError("type=1或type=2时不应该出现chord")
                        elif not note.isRest:
                            for i in range(start_time, end_time):
                                xml_list[i] = note.pitch.frequency
                        else:
                            pass

                        info_dict = {"start_point": start_time, "end_point": end_time, "note_id": note.id}

                        try:
                            if note.isChord:
                                info_dict["freq"] = note.notes[self.judge_staff - 1].pitch.frequency
                            else:
                                info_dict["freq"] = note.pitch.frequency
                        except AttributeError:
                            info_dict["freq"] = 0
                        notes_dict.append(info_dict)

        elif self.msc_type == 3:
            # 目前对于只有chord的乐谱只支持把所有音符按顺序一个一个拿出来唱
            part = self.msc.getElementsByClass("Part")[0]
            if len(self.msc.getElementsByClass("Part")) > 1:
                raise Warning("type3中检测到了含多个part的chord xml 请修改代码。本次只取第一个part分析。")
            total_quarter_length = part.quarterLength
            total_time = int(((total_quarter_length * 60) / self.speed) * 100)
            xml_list = [0 for _ in range(total_time + 1)]
            now_length = 0
            note_count = 1

            if len(part.getElementsByClass("Measure")) > 1:
                raise Warning("type3中检测到了含多个measure的chord xml 请修改代码。本次只取第一个measure分析。")
            measure = part.getElementsByClass("Measure")[0]
            # 目前对于type = 3，不存在有rest的可能
            for chord in measure.getElementsByClass("Chord"):
                for note in chord.notes:
                    if note.quarterLength != 4:
                        continue
                    start_time = int(((now_length * 60) / self.speed) * 100)
                    # 目前让chord中的每个音符均分全音符的时间后生成list和dict
                    now_length += note.quarterLength / len(chord.notes)
                    end_time = int(((now_length * 60) / self.speed) * 100)
                    note.id = str(note_count)
                    # print(note, note.id)
                    note_count += 1
                    for i in range(start_time, end_time):
                        xml_list[i] = note.pitch.frequency
                    info_dict = {"start_point": start_time, "end_point": end_time,
                                 "note_id": note.id, "freq": note.pitch.frequency}
                    notes_dict.append(info_dict)
        elif self.msc_type == 4:
            part = self.msc.getElementsByClass("Part")[0]
            if len(self.msc.getElementsByClass("Part")) > 1:
                raise Warning("type4中检测到了含多个part的voice xml 请修改代码。本次只取第一个part分析。")
            total_quarter_length = part.quarterLength
            total_time = int(((total_quarter_length * 60) / self.speed) * 100)
            xml_list = [0 for _ in range(total_time + 1)]
            now_length = 0
            note_count = 1
            for measure in part.getElementsByClass("Measure"):
                for note in measure.getElementsByClass(["Note", "Chord", "Rest"]):
                    raise Warning(
                        "请检查本题的xml，如果不是对应题干，则出现了包含正常音符的voice，请修改代码。xml地址为%s" %
                        self.xml_path)
            for measure in part.getElementsByClass("Measure"):
                voice = measure.getElementsByClass("Voice")[self.judge_staff - 1]
                for note in voice.getElementsByClass(["Note", "Rest"]):
                    start_time = int(((now_length * 60) / self.speed) * 100)
                    # 目前让chord中的每个音符均分全音符的时间后生成list和dict
                    now_length += note.quarterLength
                    end_time = int(((now_length * 60) / self.speed) * 100)
                    note.id = str(note_count)
                    note_count += 1
                    for i in range(start_time, end_time):
                        if note.isRest:
                            xml_list[i] = 0
                        else:
                            xml_list[i] = note.pitch.frequency
                    info_dict = {"start_point": start_time, "end_point": end_time,
                                 "note_id": note.id}
                    if note.isRest:
                        info_dict["freq"] = 0
                    else:
                        info_dict["freq"] = note.pitch.frequency
                    notes_dict.append(info_dict)
        elif self.msc_type == 5:
            pass
        elif self.msc_type == 6:
            for part in self.msc.getElementsByClass("Part"):
                # print("new p ")
                for measure in part.getElementsByClass("Measure"):
                    for note in measure.getElementsByClass("Note"):
                        # print(note)
                        pass
                    for voice in measure.getElementsByClass("Voice"):
                        for note in voice.getElementsByClass("Note"):
                            # print(note)
                            pass
        self.start_mute = next((i for i, x in enumerate(xml_list) if x), 0) * 10
        self.xml_list = xml_list
        self.note_dict = notes_dict
        return xml_list, notes_dict

    def get_zero(self):
        y, sr = librosa.load(self.audio_path)
        win_size = 512
        hop_size = 256
        rms = librosa.feature.rmse(y=y, frame_length=win_size, hop_length=hop_size).T
        t1 = np.arange(y.size) / sr
        t2 = np.arange(rms.shape[0]) * hop_size / sr
        # 计算端点
        flag = 0
        intervals = np.empty(shape=[0, 2])
        i0 = -1  # 起始点
        i1 = -1  # 结束点
        vth = max(rms) / 10
        for i in range(rms.shape[0]):
            if i0 < 0:  # i0为-1时开始搜索起始点
                if rms[i] > vth:
                    i0 = i * hop_size / sr
            else:  # 否则搜索结束点
                if rms[i] < vth:
                    for j in range(50):  # 判断声音幅度小于阈值之后是否有一段时间无人说话
                        if rms[i + j] < vth:
                            flag = 0
                        else:
                            flag = 1
                            break
                    if flag == 0:
                        i1 = i * hop_size / sr  # 一旦找到结束点，将起始点与结束点加入interval，并且将起始点设为-1,以便寻找下一个起始点
                        intervals = np.append(intervals, [[i0, i1]], axis=0)
                        i0 = -1
                    else:
                        continue
        return intervals

    def get_note2(self):
        # intelvals = self.get_zero()
        notes2 = []
        note_id = 1
        start_point = 0
        end_point = 0
        for i in range(len(self.voice_list) - 1):
            freq = self.voice_list[i][1]
            # print(freq)
            if abs(self.voice_list[i][1] - self.voice_list[i + 1][1]) > 18:
                end_point = self.voice_list[i][0]
                info_dict = {"start_point": start_point, "end_point": end_point,
                             "note_id": note_id, "freq": freq}
                notes2.append(info_dict)
                note_id += 1
                start_point = self.voice_list[i + 1][0]
            elif i == (len(self.voice_list) - 1):
                end_point = self.voice_list[i + 1][0]
                info_dict = {"start_point": start_point, "end_point": end_point,
                             "note_id": note_id, "freq": freq}
                notes2.append(info_dict)
        # print(notes2)
        self.note_dict2 = notes2
        return notes2

    def correct_octave_bias(self):
        # 返回相对于xml_list, voice_list的粗糙半音偏差
        xml_arr_orig = np.array(self.xml_list.copy())
        voice_arr_orig = np.array(self.voice_list.copy())
        xml_arr = xml_arr_orig[xml_arr_orig > 0]
        voice_arr = voice_arr_orig[voice_arr_orig > 0]
        xml_mean = np.mean(xml_arr)
        voice_mean = np.mean(voice_arr)
        xml_med = np.median(xml_arr)
        voice_med = np.median(voice_arr)
        log_bias = round((np.log(voice_mean / xml_mean) / np.log(2) + np.log(voice_med / xml_med) / np.log(2)) / 2)
        voice_arr = voice_arr_orig * (2 ** (-log_bias))
        self.voice_list = voice_arr.tolist()

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

    @staticmethod
    def DTW_visualize(match_dict, xml_data_dict):
        plt.cla()
        for i in range(len(match_dict)):
            if (i % 2 == 0):
                clr = 'red'
            else:
                clr = 'green'
            note_info = xml_data_dict[i]
            ft = note_info['freq']
            ftLow, ftHigh = Judge.ft2ftrange(ft)
            start_point = note_info['start_point']
            end_point = note_info['end_point']
            if (start_point == end_point): continue
            X = []
            Y1 = []
            Y2 = []
            Y3 = []
            for j in range(start_point, end_point):
                X.append(j)
                Y1.append(ftLow)
                Y2.append(ftHigh)
            Y3 = match_dict[i]['match_list'].copy()
            plt.scatter(X, Y1, color=clr, alpha=0.3)
            plt.scatter(X, Y2, color=clr, alpha=0.3)
            if (len(Y3) < len(X)):
                while (True):
                    Y3.append(Y3[-1])
                    if (len(Y3) == len(X)): break
            elif (len(Y3) > len(X)):
                while (True):
                    X.append(X[-1] + 1)
                    if (len(Y3) == len(X)): break
            plt.scatter(X, Y3, color='blue', alpha=0.3)
        plt.show()

    def DTW3(self):
        name = self.audio_path.split("/")[-1].split("\\")[-1].split(".")[0]
        path = os.path.join(self.txt_save_dir, name + ".pickle")
        # print(path)
        try:
            print(1 / 0)
            f = open(path, "rb")
            res, distance = pickle.load(f)
            f.close()
        except Exception:
            f = open(path, "wb")
            HWD = HandWriteDtw(voice_list=self.voice_list, xml_list=self.xml_list, note_dict=self.note_dict,
                               bpm=self.speed, quarterLength=self.quarterLength)
            res = HWD.DTW()
            distance = HWD.distance
            pickle.dump((res, distance), f)
            f.close()
        # HWD.NOTE_visualize()
        self.distance = distance
        print("distance", distance)
        return res

    def get_sing_score(self):
        def note_pitch_score(len_ans: int, len_user, ans_freq: float, pitch_array: [float], note_addr):
            ratio_approx = len(self.voice_list) / len(self.xml_list)
            # print(pitch_array)
            # 对pitch_array 的0值修补一下
            if np.sum(pitch_array) > 0:
                pitch_array_median = np.median([[i for i in pitch_array if i > 20]])
                pitch_array = [i if i > 20 else pitch_array_median for i in pitch_array]
            # print(1111111111, len_user, len_ans, ans_freq)
            # plt.clf()
            # # print(111111111111)
            # plt.scatter([i for i in range(len(pitch_array))], pitch_array)
            # plt.axhline(y=ans_freq)
            # plt.show()
            # print("得到的数据", len_ans, len_user, ans_freq, pitch_array)
            text = ""
            # if len(pitch_array) < 6 * ratio_approx:
            #     print("传过来的数据长度竟然只有%s？按bpm=120的常规速度那就是%s秒, 小于一个三十二分音符的时长，一定是匹配错了,返回10分"
            #           % (len(pitch_array), len(pitch_array) * 0.01))
            #     self.single_note_judge.append(
            #         {'score': 10, 'delta_key': np.log(np.mean(pitch_array) / ans_freq) / np.log((2 ** (1 / 12))), 'user_audio_length': len_user,
            #          'ans_audio_length': len_ans, 'text': '时间过短检测不到。\n', 'note_addr': note_addr})
            #     return {'score': 10, 'text': "时间过短检测不到。\n", 'user_audio_length': len_user, 'ans_audio_length': len_ans}
            assert ans_freq > 0, "所给题目频率小于等于零！"
            assert len_user > 0, "用户音频长度为零！"
            # if len_user / len_ans < 0.15:
            #     print("用户音频不足标准答案15%，怀疑是数据错误匹配，返回10分")
            #     self.single_note_judge.append({'score': 10, 'delta_key': np.log(np.mean(pitch_array) / ans_freq) / np.log((2 ** (1 / 12))), 'user_audio_length': len_user,
            #                                    'ans_audio_length': len_ans, 'text': '时间过短怀疑是算法匹配错误。\n', 'note_addr': note_addr})
            #     return {'score': 10, 'text': "时间过短怀疑是算法匹配错误。\n", 'user_audio_length': len_user, 'ans_audio_length': len_ans}
            if len_user / len_ans < 0.5 * ratio_approx:
                text += "时长略短，"
            elif len_user / len_ans > 2 * ratio_approx:
                text += "时长略长，"
            np_pitch_array = np.array(pitch_array)
            pitch_array_max = np.max(np_pitch_array)

            head_ratio = 0.12
            tail_ratio = 0.1
            # 归一化数组和频率
            standard_pitch_array = np_pitch_array - np.min(np_pitch_array)
            standard_pitch_array = standard_pitch_array / (
                1 if np.max(standard_pitch_array) == 0 else np.max(standard_pitch_array))
            standard_ans_freq = ans_freq / np.max(standard_pitch_array)
            array_var = np.var(standard_pitch_array)
            array_mean = np.mean(standard_pitch_array)
            array_head = standard_pitch_array[:round(len_user * head_ratio)]
            # array_body = standard_pitch_array[round(len_user * head_ratio): round(len_user * (1 - tail_ratio))]
            array_body = standard_pitch_array
            body_pitch = np.median(np_pitch_array[round(len_user * head_ratio): round(len_user * (1 - tail_ratio))])
            array_tail = standard_pitch_array[round(len_user * (1 - tail_ratio)):]

            epsilon = 1e-9
            # ##################################主体检测#######################################
            body_mean = np.mean(array_body)

            # 相差的半音
            # delta_key = (abs(standard_ans_freq / (body_mean + epsilon)) - 1) / (2 ** (1 / 12))
            delta_key = np.log(body_pitch / ans_freq) / np.log((2 ** (1 / 12))) if ans_freq > 0 else 0
            print("##############", delta_key, body_pitch, ans_freq, np_pitch_array)
            # 主体抖动程度，方差最大为0.25最小为0
            body_var = np.var(array_body)
            # 主体音高扣分
            if abs(delta_key) <= 0.2:
                pitch_error = 0
            elif 0.7 >= abs(delta_key) > 0.2:
                pitch_error = (abs(delta_key) - 0.2) * 30
            elif 1.2 >= abs(delta_key) > 0.7:
                pitch_error = (abs(delta_key) - 0.2) * 30
                text += "音高有1key左右偏差，"
            elif 2.3 >= abs(delta_key) > 1.2:
                pitch_error = 30 + (abs(delta_key) - 1.4) * 40
                text += "音高有2key左右偏差，"
            elif 3.2 >= abs(delta_key) > 2.3:
                pitch_error = 70 + (abs(delta_key) - 2.3) * 25
                text += "音高有2key左右偏差，"
            else:
                pitch_error = 95
                text += "音高有较大偏差，"

            # 主体稳定性扣分
            print(abs(body_var))
            body_stability_error = min(abs(body_var) * 90 / 0.18, 100)
            # if body_stability_error < 15:
            #     text += "稳定程度较好。\n"
            # elif body_stability_error < 50:
            #     text += "稳定程度一般。\n"
            # else:
            #     text += "稳定程度较差。\n"
            text += "\n"
            # print(body_var, body_stability_error, text, end="")
            # 主体扣分
            body_error = pitch_error * 0.6 + body_stability_error * 0.4

            # ################################################################################
            # ##################################头部检测#######################################
            # 头部稳定性检测
            head_var = np.var(array_head)
            if np.isnan(head_var):
                head_var = 0
            head_stability_error = abs(head_var - 0.06) * 90

            # 倚音检测待添加

            head_error = head_stability_error
            # ################################################################################

            # ##################################尾部检测#######################################
            # 尾部稳定性检测
            tail_var = np.var(array_tail)
            if np.isnan(tail_var):
                tail_var = 0
            tail_stability_error = max(0, tail_var - 0.1) * 90

            tail_error = tail_stability_error
            # ################################################################################
            # errscore = int(head_error * head_ratio + body_error * (1 - head_ratio - tail_ratio) + tail_error * tail_ratio)
            errscore = body_error
            score = 100 - errscore
            # if len_user < len_ans * 0.6:
            #     score *= len_user / (len_ans * 0.6)
            # elif len_user * 0.6 > len_ans:
            #     score *= max(0, len_ans / (len_user * 0.3) - 1)
            score = int(score)
            # print(head_error, body_error, tail_error)
            self.single_note_judge.append({'score': score, 'delta_key': delta_key, 'user_audio_length': len_user,
                                           'ans_audio_length': len_ans, 'text': text, 'note_addr': note_addr})
            return {'score': score, 'delta_key': delta_key, 'user_audio_length': len_user,
                    'ans_audio_length': len_ans, 'text': text}

        def color(uncolored_note):
            # 上色
            this_note_score = uncolored_note.pitch_score
            if this_note_score > 75:
                uncolored_note.style.color = "#008000"
            elif this_note_score > 60:
                uncolored_note.style.color = "#b0b500"
            else:
                uncolored_note.style.color = "#ff0000"
            # 为每个音符添加分数
            # uncolored_note.lyric = this_note_score

        def plot_match_dict(match_dict):
            note_dict = self.note_dict
            A = 440
            num = 0
            print(len(match_dict), len(note_dict))
            for i in range(-12, 13):
                plt.axhline(y=i, ls="-", c="red", alpha=0.1)
            for i, v in enumerate(match_dict):
                plt.axvline(x=num, ls="-", c="green", alpha=0.2)
                # print(note_dict[i]['freq'])
                rel_pitch_match = [np.log(s / A) / np.log(2) * 12 for s in v['match_list']]
                rel_pitch_note = np.log(note_dict[i]['freq'] / A) / np.log(2) * 12
                len_rel_pitch_match = len(rel_pitch_match)
                plt.scatter([num + s for s in range(len_rel_pitch_match)], rel_pitch_match, c='r' if i % 2 else 'g',
                            s=1)
                plt.scatter([num + s for s in range(len_rel_pitch_match)],
                            [rel_pitch_note for _ in range(len_rel_pitch_match)],
                            c='black', s=4, alpha=0.1)
                num += len(rel_pitch_match)
            ax = plt.axes()
            ax.set_yticks([i for i in range(-24, 25)])
            ax.set_yticklabels(
                [["C", "", "D", "", "E", "F", "", "G", "", "A", "", "B"][(i + 9) % 12] for i in range(-24, 25)])
            plt.show()

        def get_ratio(match, k):
            ratios = []
            offsets = []
            for i in range(len(self.note_dict)):
                # note = ctypes.cast(self.note[i], ctypes.py_object).value
                note = self.note[i]
                offsets.append(note.offset)
            # print(offsets)
            count = 0
            begin = self.note_dict[1]['start_point']
            end = 0
            begin_voice = match[1]['start_point']
            end_voice = 0
            # print(offsets)
            for i in range(1, len(self.note_dict) - 1):
                if i == len(self.note_dict) - 2:
                    end = float(self.note_dict[i - 1]['end_point'] + 1)
                    end_voice = match[i - 1]['end_point'] + 1
                    m = (end - begin) / (end_voice - begin_voice)
                    for _ in range(count + 1):
                        ratios.append(m)
                    if offsets[-1] == 0:
                        ratios.append(ratios[-1])
                    if offsets[1] == 0:
                        ratios = [ratios[0]] + ratios
                    return ratios
                if abs(offsets[i] - 0) < 0.1:
                    count += 1
                    if count == k:
                        end = float(self.note_dict[i - 1]['end_point'])
                        end_voice = match[i - 1]['end_point']
                        if end_voice - begin_voice == 0:
                            count -= 1
                            continue
                        m = (end - begin) / (end_voice - begin_voice)
                        for _ in range(k):
                            ratios.append(m)
                        count = 0
                        begin = self.note_dict[i]['start_point']
                        begin_voice = match[i]['start_point']

        all_zero = 1
        for point in self.voice_list:
            if point > 0:
                all_zero = 0
                break
        if all_zero:
            return "all_zero"
        vaild_length = 0
        score_mult_time = 0
        self.correct_octave_bias()
        try:
            match_dict = self.DTW3()
        except AssertionError:
            self.score = 0
            self.text = "演唱时长过短或检测不到录音，请回放音频检查并重新录制！有问题可申请人工复审。"
            return
        self.k_measure_ratio = get_ratio(match_dict, 1 if self.beats > 3 else 2)
        # if self.distance > 5000:
        #     self.score = self.pitch_score = -1
        #     self.text = "演唱错误过多，请重新演唱或上传人工复审。"
        #     return
        print(get_ratio(match_dict, 2))
        # plot_match_dict(match_dict)
        index = 0
        number = 1
        final_text = ""
        if self.msc_type == 1 or self.msc_type == 2:
            for part in self.msc.getElementsByClass("Part"):
                if number != self.judge_staff:
                    number += 1
                    continue
                number += 1
                for measure in part.getElementsByClass("Measure"):
                    while True:
                        if index == len(match_dict):
                            break
                        note_id = match_dict[index]["note_id"]
                        match_list = match_dict[index]["match_list"]
                        note = measure.getElementById(note_id)
                        # print("note:", note, note_id, type(note_id))
                        # print(note_id, end= " ")
                        if note:
                            note_txt = f"第{note_id}个音符"
                            len_ans = int(((note.quarterLength * 60) / self.speed) * 100)
                            if len_ans == 0:
                                print("这里是倚音，暂不处理")
                                index += 1
                                continue
                            len_user = len(match_list)
                            try:
                                ans_freq = note.pitch.frequency
                            except AttributeError:
                                index += 1
                                continue
                            # print(index)
                            note.pitch_score_dict = note_pitch_score(len_ans, len_user, ans_freq, match_list, id(note))
                            note.pitch_score = note.pitch_score_dict["score"]
                            if len(note.pitch_score_dict["text"]) > 1:
                                final_text += note_txt + note.pitch_score_dict["text"]
                            score_mult_time += note.pitch_score * note.quarterLength
                            vaild_length += note.quarterLength
                            color(note)
                        else:
                            break
                        index += 1
        elif self.msc_type == 3:
            part = self.msc.getElementsByClass("Part")[0]
            measure = part.getElementsByClass("Measure")[0]
            for chord in measure.getElementsByClass("Chord"):
                # for note in chord.notes:
                #     print(note)
                for note in chord.notes:
                    if note.quarterLength != 4:
                        continue
                    for index in range(100):

                        if match_dict[index]["note_id"] == note.id:
                            final_text += f"第{match_dict[index]['note_id']}个音符"
                            match_list = match_dict[index]["match_list"]
                            len_ans = int(((note.quarterLength * 60) / self.speed) * 100)
                            len_user = len(match_list)
                            ans_freq = note.pitch.frequency
                            note.pitch_score_dict = note_pitch_score(len_ans, len_user, ans_freq, match_list, id(note))
                            note.pitch_score = note.pitch_score_dict["score"]
                            final_text += note.pitch_score_dict["text"]
                            score_mult_time += note.pitch_score * note.quarterLength
                            vaild_length += note.quarterLength
                            color(note)
                            break
        elif self.msc_type == 4:
            part = self.msc.getElementsByClass("Part")[0]
            for measure in part.getElementsByClass("Measure"):
                voice = measure.getElementsByClass("Voice")[self.judge_staff - 1]
                for note in voice.getElementsByClass(["Note", "Rest"]):
                    if match_dict[index]["note_id"] == note.id:
                        final_text += f"第{match_dict[index]['note_id']}个音符"
                        match_list = match_dict[index]["match_list"]
                        len_ans = int(((note.quarterLength * 60) / self.speed) * 100)
                        len_user = len(match_list)
                        try:
                            ans_freq = note.pitch.frequency
                        except AttributeError:
                            # 休止符
                            pass
                        note.pitch_score_dict = note_pitch_score(len_ans, len_user, ans_freq, match_list, id(note))
                        note.pitch_score = note.pitch_score_dict["score"]
                        final_text += note.pitch_score_dict["text"]
                        score_mult_time += note.pitch_score * note.quarterLength
                        vaild_length += note.quarterLength
                        color(note)
                        index += 1
        self.pitch_score = int(score_mult_time / vaild_length)
        self.score += self.pitch_score
        self.text = final_text
        self.global_sing_score()
        # self.draw_seg_wav()

    def draw_seg_wav(self):
        y, sr = librosa.load(self.audio_path, sr=44100)
        plt.clf()
        plt.plot([i for i in range(len(y))], y)
        cut_points = [51, 77, 103, 129, 155, 206, 229, 254, 273, 299, 351, 376, 413, 439, 465, 491, 517, 543, 568, 620,
                      646, 659,
                      672, 685, 711, 762, 784, 825, 877, 903, 929, 954, 980, 1032, 1056, 1082, 1097, 1122, 1174, 1200,
                      1263, 1289,
                      1315, 1340, 1366, 1392, 1418, 1470, 1496, 1550, 1575, 1601, 1705, 1756, 1782, 1808, 1860, 1899,
                      1912, 1937,
                      1963, 1989, 2015, 2024, 2025, 2068, 2120, 2146, 2172, 2224, 2262, 2275, 2299, 2311, 2324, 2337,
                      2363, 2415,
                      2437, 2481, 2533, 2559, 2585, 2611, 2636, 2684, 2708, 2734, 2751, 2777, 2828, 2850, 2893, 2919,
                      2945, 2971,
                      2997, 3022]
        txt = ''
        ind = 0
        for i in cut_points:
            ind += 1
            plt.axvline(x=i * 441, c='r')
            txt += '''  TRACK 0%s AUDIO
    TITLE "音符"
    INDEX 01 00:%s:%s\n''' % (ind + 1, str(i // 100).zfill(2), str(i)[-2:])
        print(txt)
        plt.show()
        pass

    def global_sing_score(self):
        # 一、处理节奏
        # 1.第一步， 把人声速度和乐曲速度作一下统一
        voice_point = [v['user_audio_length'] for v in self.single_note_judge]
        xml_point = [v['ans_audio_length'] for v in self.single_note_judge]
        speed_ratio = [v['ans_audio_length'] / v['user_audio_length'] for v in self.single_note_judge]
        speed_ratio_median = np.median(speed_ratio)
        print("voice_point:%s, xml_point:%s" % (len(voice_point), len(xml_point)))
        print("sum_voice_point:%s, sum_xml_point:%s" % (sum(voice_point), sum(xml_point)))
        delta_keys = [v['delta_key'] for v in self.single_note_judge]
        global_delta_keys = round(np.median(delta_keys) if delta_keys else 0)
        delta_keys_std = [v['delta_key'] - global_delta_keys for v in self.single_note_judge]
        speed_ratio_std = np.array(speed_ratio) / speed_ratio_median
        # global_speed_ratio = np.mean(speed_ratio_std)
        global_speed_ratio = sum(xml_point) / sum(voice_point)
        print("速度折合比率为%s" % global_speed_ratio, "音高折合比率为%s" % global_delta_keys)
        # 比率大于5/4即判定为不准确
        # TODO 长音/换气点可能唱不满
        # 2. 标记音高错误
        pitch_start = -1
        pitch_end = -1
        high = True
        for i in range(len(self.single_note_judge)):
            if abs(delta_keys_std[i]) > 0.5:
                high = delta_keys_std[i] > 0
                if pitch_start == -1:
                    pitch_start = i
            else:
                if pitch_start != -1:
                    pitch_end = i - 1
                    if pitch_start == pitch_end:
                        note = ctypes.cast(self.single_note_judge[pitch_start]['note_addr'], ctypes.py_object).value
                        note.addLyric("↑" if high else "↓", 2)
                        # print(type(note))
                    else:
                        note = ctypes.cast(self.single_note_judge[pitch_start]['note_addr'], ctypes.py_object).value
                        note.addLyric("{" + ("↑" if high else "↓"), 2)
                        note = ctypes.cast(self.single_note_judge[pitch_end]['note_addr'], ctypes.py_object).value
                        note.addLyric("}", 2)
                    pitch_start = pitch_end = -1
        # 3. 标记节奏错误

        # rhythm_start = -1
        # rhythm_end = -1
        # fast = 0
        # for i in range(len(self.single_note_judge)):
        #     # noTe = ctypes.cast(self.single_note_judge[i]['note_addr'], ctypes.py_object).value
        #     # music21.note.Note().addLyric(, 2)
        #     if not 1.25 > speed_ratio_std[i] > 0.8:
        #         fast += int(speed_ratio_std[i] > 1)
        #         if rhythm_start == -1:
        #             rhythm_start = i
        #     else:
        #         if rhythm_start != -1:
        #             if not 1.5 > speed_ratio_std[i - 1] > 0.666:
        #                 continue
        #             else:
        #                 rhythm_end = i - 1
        #                 fast = fast / (rhythm_end - rhythm_start + 1)
        #
        #                 if fast < 0.33:
        #                     lyric = "F"
        #                 elif fast > 0.67:
        #                     lyric = 'S'
        #                 else:
        #                     lyric = 'U'
        #                 print(fast, (rhythm_end - rhythm_start + 1), lyric)
        #                 if rhythm_start == rhythm_end:
        #                     note = ctypes.cast(self.single_note_judge[rhythm_start]['note_addr'], ctypes.py_object).value
        #                     note.addLyric(lyric, 3)
        #
        #                     # print(type(note))
        #                 else:
        #                     note = ctypes.cast(self.single_note_judge[rhythm_start]['note_addr'], ctypes.py_object).value
        #                     note.addLyric("[" + lyric, 3)
        #                     note = ctypes.cast(self.single_note_judge[rhythm_end]['note_addr'], ctypes.py_object).value
        #                     note.addLyric("]", 3)
        #                 rhythm_start = rhythm_end = -1
        #                 fast = 0

        # 4.尝试按小节标记节奏错误
        for ind, part in enumerate(self.msc.getElementsByClass("Part")):
            if ind != self.judge_staff - 1:
                continue
            for m_ind, measure in enumerate(part.getElementsByClass("Measure")):
                first_note = measure.getElementsByClass(["Note", "Chord", "Rest"])[0]

                # measure_speed_ratio = self.k_measure_ratio[m_ind] / np.sqrt(global_speed_ratio)
                measure_speed_ratio = 1
                if 1.11 <= measure_speed_ratio < 1.25:
                    first_note.lyric = ">"
                    self.score -= 1
                elif 1.25 <= measure_speed_ratio:
                    first_note.lyric = ">>"
                    self.score -= 2
                elif 0.9 >= measure_speed_ratio > 0.8:
                    first_note.lyric = "<"
                    self.score -= 1
                elif 0.8 >= measure_speed_ratio:
                    first_note.lyric = "<<"
                    self.score -= 2
        # print("!!!", global_speed_ratio)
        if 0.9 < global_speed_ratio < 1.11:
            self.text = "整体节奏稳定。\n" + self.text
        elif 1.11 <= global_speed_ratio < 1.25:
            self.text = "整体节奏稍快。\n" + self.text
            self.score -= 5
        elif 0.8 < global_speed_ratio <= 0.9:
            self.text = "整体节奏稍慢。\n" + self.text
            self.score -= 5
        elif global_speed_ratio >= 1.25:
            self.text = "整体节奏过快。\n" + self.text
            self.score -= 15
        elif 0.8 < global_speed_ratio <= 0.8:
            self.text = "整体节奏过慢。\n" + self.text
            self.score -= 5

        if -1 < global_delta_keys < 1:
            self.text = "整体音高稳定。\n" + self.text
        elif 1 <= global_delta_keys:
            self.text = f"整体偏高{global_delta_keys}个半音。\n" + self.text
            self.score -= 5 * global_delta_keys
        elif -1 >= global_delta_keys:
            self.text = f"整体偏低{-global_delta_keys}个半音。\n" + self.text
            self.score -= 5 * global_delta_keys
        # 画图
        # plt.clf()
        # plt.title("key_bias:%s, rhythm_bias:%s" % (global_delta_keys, np.mean(speed_ratio_std)))
        # plt.ylim([-2, 2])
        # plt.scatter(np.arange(0, len(speed_ratio_std)), speed_ratio_std, color=["b" if 0.8 < i < 1.25 else "r" for i in speed_ratio_std])
        # plt.scatter(np.arange(0, len(delta_keys_std)), delta_keys_std, color=["b" if -0.5 < i < 0.5 else "r" for i in delta_keys_std], marker="+")
        # plt.show()

        return 0

    def get_total_score(self):
        if self.get_sing_score():
            return "none_sing_score"

    def cut_xml_by_line(self, line):
        msc = copy.deepcopy(self.msc)
        for part in msc.getElementsByClass("Part"):
            key = part.getElementsByClass("Measure")[0].getElementsByClass("Key")[0]
            for measure in part.getElementsByClass("Measure"):
                if measure.line != line:
                    part.pop(part.index(measure))
            first_measure = part.getElementsByClass("Measure")[0]
            if not first_measure.getElementsByClass("Key"):
                first_measure.insert(0, key)
        return msc

    def save_data_to_txt(self, data):
        import time
        save_path = os.path.join(self.txt_save_dir, "%s.txt" % str(time.time()).split(".")[0])
        import json
        with open(save_path, "w", encoding="utf-8") as f:
            save_data = json.dumps(data)
            f.write(save_data)
            f.close()
        return save_path

    def export(self):
        # if True:
        #     metronome_abspath = os.path.join(self.txt_save_dir, "metronome_" + self.audio_path.split("/")[-1].split("\\")[-1])
        #     metronome_relpath = os.path.relpath(metronome_abspath, os.path.abspath(".")).replace("\\", "/")
        #     res_dict = {"metronome_audio_path": metronome_relpath if os.path.exists(metronome_relpath)
        #                 else "library" + self.audio_path.split("library")[1],
        #                 "wav"
        #                 "start": self.start_point - self.start_mute}
        #     save_path = self.save_data_to_txt(res_dict)
        #     return save_path, 0
        if self.get_total_score():
            res_dict = {"score": 0, "png_x_info": [], "text": "演唱时长不足或录制音频音量为0，如多次出现请进行反馈！",
                        "start": self.start_point - self.start_mute}
            save_path = self.save_data_to_txt(res_dict)
            return save_path, 0
        next_start = 0
        line = 1
        import time
        time_name = str(time.time()).split(".")[0]
        pngstr = ""
        png_x_info = []
        rst_x, score = [], self.score  # 预设待改
        print(self.score, self.pitch_score)
        for line in range(1, self.total_line + 1):
            try:
                part_xml = self.cut_xml_by_line(line=line)
                file_name = os.path.join(self.txt_save_dir,
                                         "%s_%s_%s.xml" % (self.xml_path.split("/")[-1].split(".")[0], time_name, line))
                fp = part_xml.write("musicxml", fp=file_name)

                pngPath = part_xml.write("musicxml.png", fp=fp)
                # TODO 假装有图片
                # pngPath = "musicxml.png"

                pngPath = os.path.relpath(pngPath, os.path.abspath("."))
                pngstr += pngPath + "*"
                print(pngPath)
                png_x_info.append({"X": rst_x, "pngPath": pngPath})
            except IndexError:
                print("这应该是双声部的问题。停止循环。")
                break
        # stm.show()

        # pngPath = self.xml_path
        # pngPath = pngPath[:pngPath.rfind("\\")] + pngPath[pngPath.rfind("\\"):pngPath.rfind(".")] + ".png"

        # res_dict = {"score": score, "pitch_data": xmlftrange, "png_x_info": png_x_info}
        res_dict = {"score": score, "png_x_info": png_x_info}
        if len(png_x_info) == 0 or score == 0:
            res_dict["text"] = "演唱时长过短或检测不到录音，请回放音频检查并重新录制！有问题可申请人工复审。"
        elif 0 < score < 60:
            res_dict["text"] = "演唱并非本乐曲或演唱与标准答案相差太大！请重新演唱或提交人工复审。"
        else:
            res_dict["text"] = self.text
        print(self.text)

        metronome_abspath = os.path.join(self.txt_save_dir,
                                         "metronome_" + self.audio_path.split("/")[-1].split("\\")[-1])
        metronome_relpath = os.path.relpath(metronome_abspath, os.path.abspath(".")).replace("\\", "/")
        res_dict["metronome_audio_path"] = metronome_relpath if os.path.exists(metronome_relpath) else os.path.relpath(
            self.audio_path, os.path.abspath(".")).replace("\\", "/")
        save_path = self.save_data_to_txt(res_dict)

        return save_path, score


if __name__ == "__main__":
    j = Judge(audio_path=r'001000000000000020101-48.mp3',
              xml_path=r'3_课次1_1_28_72.xml',
              txt_save_dir=r'melodyExtraction_JDC/results',
              staff_choice=1)
    j.get_xml_list()
    j.get_voice_list()
    j.correct_octave_bias()
    # print(j.xml_list)
    # print(j.voice_list)
    # print(time.time())
    j.export()
