import matplotlib.pyplot as plt

def ftt_plt():
    t = []
    ft = []
    f = open("./results/pitch_test_audio_file.mp3.txt", "r")
    line = f.readline()
    line = line[:-1]
    line = str(line)
    t1, ft1 = line.split()
    t.append(float(t1)), ft.append(float(ft1))
    while line:
        line = f.readline()
        line = line[:-1]
        line = str(line)
        if (len(line) == 0): break
        t1, ft1 = line.split()
        t.append(float(t1)), ft.append(float(ft1))
    f.close()

    plt.plot(t, ft)
    plt.show()
