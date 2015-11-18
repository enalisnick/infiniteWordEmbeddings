import os
from os import listdir
from os.path import isfile, join

logspath = "/Users/enalisnick/Desktop/iSG/logs/"
vecspath = "/Users/enalisnick/Desktop/vectors/"
picname = "p_z_w_for_bat_rock_apple_race.png"
# files = [ f for f in listdir(logspath) if isfile(join(logspath,f)) ]
# only use the ones that looked good
files = []
roots = []
for file in files:
    roots.append(file.split('PROGRESS_')[1].split('.log')[0])
roots = set(roots)
for root in roots:
    os.system("python graph_p_z.py -i %s -c %s -w bat,rock,apple,race" %(vecspath+'INPUT_VECS_'+root+'.txt', vecspath+'CONTEXT_VECS_'+root+'.txt'))
    os.system("mv %s %s " %(picname, root+"-"+picname))
