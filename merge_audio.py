from pydub import AudioSegment
import os


def start_detection(audio_path):
    return 3000.0


def match_target_amplitude(audio_path, target_dBFS=-20):
    sound = AudioSegment.from_mp3(audio_path)
    change_in_dBFS = target_dBFS - sound.dBFS
    normalized_sound = sound.apply_gain(change_in_dBFS)
    normalized_sound.export(audio_path, format="mp3")


def merge_audio(part1_audio_path, part2_audio_path, judge_part, delay, save_dir):
    # 把sound2覆盖在sound1上，两个音频文件会叠加，如果sound2较长，则会被截断。
    sound1 = AudioSegment.from_mp3(part1_audio_path)
    sound2 = AudioSegment.from_mp3(part2_audio_path)
    delay = delay-120
    if str(judge_part) == "1":
        output_name = part1_audio_path.split("/")[-1].split("\\")[-1]
        if delay > 0:
            output = sound2.overlay(sound1, position=delay)
        else:
            output = sound1.overlay(sound2, position=abs(delay))
    # mix sound2 with sound1, starting at 0ms into sound1)
    else:
        output_name = part2_audio_path.split("/")[-1].split("\\")[-1]
        if delay > 0:
            output = sound1.overlay(sound2, position=delay)
        else:
            output = sound2.overlay(sound1, position=abs(delay))
    output_name = "mixed_" + output_name
    # save the result
    save_path = os.path.join(save_dir, output_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output.export(save_path, format="mp3")
    return save_path


def add_metronome(audio_path, save_dir, beat_type, speed, start_point, metronome_delay=500):
    print(audio_path, save_dir, beat_type, speed, start_point)
    output_name = "metronome_" + audio_path.split("/")[-1].split("\\")[-1]
    save_path = os.path.join(save_dir, output_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not start_point:
        start = start_detection(audio_path)
    else:
        start = int(start_point)
    sound1 = AudioSegment.from_mp3(audio_path)
    base_sound = AudioSegment.from_mp3(r"./library/empty.mp3")
    metronome = AudioSegment.from_mp3(r'./library/metronome/%s_%s.mp3' % (beat_type, speed))

    base_sound = base_sound[:(len(sound1) + len(metronome) - int(start)) + 1]
    start_ms = max(500, len(metronome) - int(start))
    metronome_base = base_sound.overlay(metronome)
    if start > len(metronome):
        sound = sound1[start - len(metronome):]
        output = metronome_base.overlay(sound)
    else:
        output = metronome_base.overlay(sound1, start_ms-500)
    output.export(save_path, format="mp3")
    return save_path


# add_metronome(audio_path=r'library/usersong/owHu-4hmasv4_DxR-SC8PRVr48DM/owHu-4hmasv4_DxR-SC8PRVr48DM_003000000000100030101_1615617448853981633.mp3',
#               save_dir =r'library\user_upload_ques\owHu-4hmasv4_DxR-SC8PRVr48DM', beat_type=4, speed=75, start_point=1200)


if __name__ == "__main__":
    # add_metronome(r"D:\untitled1\library\usersong\owHu-4jpfnSsns-cm2FPMmO5OGsc\owHu-4jpfnSsns-cm2FPMmO5OGsc002000000000000010301138424.mp3",
    #               save_dir=r".", speed=120, beat_type=4)
    # merge_audio(r"D:\untitled1\library\usersong\owHu-4jpfnSsns-cm2FPMmO5OGsc\owHu-4jpfnSsns-cm2FPMmO5OGsc002000000000000010301138424.mp3",
    #             r"D:\untitled1\library\usersong\owHu-4jpfnSsns-cm2FPMmO5OGsc\owHu-4jpfnSsns-cm2FPMmO5OGsc002000000000000010301138424.mp3",
    #             1, -320, save_dir=r".")
    pass
