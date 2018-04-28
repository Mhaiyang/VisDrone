label_txt = open('../data/VisDrone2018-DET-train/annotations/0000002_00005_d_0000014.txt', 'r')

try:
    count = 0
    for line in label_txt:
        (x, y, w, h, score, cls, truncation, occlusion) = line.split(',')
        print(x, y, w, h, score, cls, truncation, occlusion)
        count += 1
finally:
    print(count)
    label_txt.close()