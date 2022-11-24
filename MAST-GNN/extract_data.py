import csv
import os

import numpy as np

lowlable = 0
midlable = 0
highlable = 0
PATH = r"C:\Users\13927\Desktop\gcn\sector_data"


def extractone(sector):
    global lowlable, midlable, highlable
    with open(sector, 'r') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        rows = []
        shouldextractfeature = [0, 1, 3, 4, 6, 7, 8, 9, 10, 13, 14, 15, 18, 20, 22, 24, 43]
        for row in reader:
            tmprow = []
            for i in shouldextractfeature:
                tmprow.append(row[i])
            lable = int(float(tmprow[-1]))
            if lable == 0:
                lowlable = lowlable + 1
            elif lable == 1:
                midlable = midlable + 1
            elif lable == 2:
                highlable = highlable + 1
            rows.append(tmprow)
    return rows


dirs = os.listdir(PATH)
alldata = []
for dir in dirs:
    shanqu = os.path.join(PATH, dir)
    shanqudirs = os.listdir(shanqu)
    singlgday = []
    for shanqudir in shanqudirs:
        sectordatapath = os.path.join(shanqu, shanqudir)
        print(sectordatapath)
        tmp = extractone(sectordatapath)
        singlgday.append(tmp)
    print("lowlable", lowlable, "midlable", midlable, "highlable", highlable)
    alldata.append(singlgday)
    temp = np.array(alldata)
    temp1 = np.array(singlgday)
    print(temp.shape, temp1.shape)
alltosave = np.array(alldata).astype(float)
alltosave = alltosave.reshape(-1, alltosave.shape[2], alltosave.shape[3]).transpose((1, 0, 2))

print(alltosave.shape)
np.save(r"./data/sector_feature.npy", alltosave)
print("save .npy done")
print("FINAL", "lowlable", lowlable, "midlable", midlable, "highlable", highlable)
# print(tmp.shape)
# print(shanqu)
# tmp=extractone("sector1.csv")
# print(tmp.shape)
