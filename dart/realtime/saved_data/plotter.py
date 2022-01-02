import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_np_array(file,samples):
    lines = file.readlines()
    n_lines = len(lines)
    data = np.zeros((n_lines,samples))
    for i,line in enumerate(lines):
        entries = line.split(' ')
        entries[-1] = entries[-1][:-1]
        cur = 0
        for e in entries:
            if e != '':
                data[i,cur] = float(e)
                cur += 1
    return data
parser = argparse.ArgumentParser()
parser.add_argument('--loss',type=str,default='Losses')
parser.add_argument('--solution',type=str,default='Solutions')
parser.add_argument('--exp_name',type=str,default=None,required=True)
parser.add_argument('--xlabel',type=str,default='mass')
parser.add_argument('--upper',type=float,default= 2.0)
parser.add_argument('--lower',type=float,default=-2.0)
args = parser.parse_args()


file = open(f'raw_data/{args.loss}.txt')
np_data = load_np_array(file,200)
file.close()
file = open(f'raw_data/{args.solution}.txt')
solutions = load_np_array(file,1)
file.close()
np.savez(f'np_files/{args.exp_name}.npz',loss=np_data,solution=solutions)

x = np.linspace(start=args.lower,stop=args.upper,num=200)
os.makedirs(f'images/{args.exp_name}',exist_ok=True)
for i in range(np_data.shape[0]):
    plt.clf()
    plt.plot(x,np_data[i])
    plt.axvline(x=float(solutions[i]),color='red')
    plt.xlabel(args.xlabel)
    plt.ylabel('loss')
    plt.savefig(f'images/{args.exp_name}/fig{i}.png',dpi=200)
