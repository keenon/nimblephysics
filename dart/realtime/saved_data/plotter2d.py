import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--loss',type=str,default='Losses')
parser.add_argument('--solution',type=str,default='Solutions')
parser.add_argument('--exp_name',type=str,default=None,required=True)
parser.add_argument('--rest_dim',type=int,default=2)
parser.add_argument('--x_upper',type=float,default=  2.0)
parser.add_argument('--x_lower',type=float,default= -2.0)
parser.add_argument('--y_upper',type=float,default=  2.0)
parser.add_argument('--y_lower',type=float,default= -2.0)
args = parser.parse_args()

np_data = []
csv_files = os.listdir(f'raw_data/{args.loss}/')
for f in csv_files:
    np_data.append(np.genfromtxt(f'raw_data/{args.loss}/{f}',delimiter=',').T)

solutions = np.genfromtxt(f'raw_data/{args.solution}.csv',delimiter=',')

np.savez(f'np_files/{args.exp_name}_2d.npz',loss=np_data,solution=solutions)

os.makedirs(f'images/{args.exp_name}',exist_ok=True)
for i in range(len(np_data)):
    plt.clf()
    mean = np_data[i].mean()
    plt.imshow(np_data[i].clip(0,mean),
               origin='lower',
               extent=[args.x_lower,args.x_upper,
                       args.y_lower,args.y_upper],
               cmap='hot')
    cold_index = [0,1,2]
    cold_index.remove(args.rest_dim)
    plt.plot(solutions[i][cold_index[0]],
             solutions[i][cold_index[1]],
             'bo')
    plt.savefig(f'images/{args.exp_name}/fig{i}.png',dpi=200)
