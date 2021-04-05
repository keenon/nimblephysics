#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:57:23 2021

@author: yannis
"""


from matplotlib import pyplot as plt
import numpy as np

filename = 'dcp.npz'
data = np.load(filename)
times = data['T_ddp']
cost = data['C_ddp']
times2 = data['T_ss']
cost2 = data['C_ss']
total_data = np.concatenate((cost, cost2), axis=0)
mind = min(total_data)
maxd = max(total_data)
cost = (cost - mind)/(maxd-mind)
cost2 = (cost2-mind)/(maxd-mind)


plt.figure()

filename = 'cp.npz'
data = np.load(filename)
times = data['T_ddp']
cost = data['C_ddp']
times2 = data['T_ss']
cost2 = data['C_ss']
total_data = np.concatenate((cost, cost2), axis=0)
mind = min(total_data)
maxd = max(total_data)
cost = (cost - mind)/(maxd-mind)
cost2 = (cost2-mind)/(maxd-mind)

plt.plot(np.array(times), np.array(cost), 'g')
plt.plot(np.array(times2), np.array(cost2), 'g--')


filename = 'dcp.npz'
data = np.load(filename)
times = data['T_ddp']
cost = data['C_ddp']
times2 = data['T_ss']
cost2 = data['C_ss']
total_data = np.concatenate((cost, cost2), axis=0)
mind = min(total_data)
maxd = max(total_data)
cost = (cost - mind)/(maxd-mind)
cost2 = (cost2-mind)/(maxd-mind)

plt.plot(np.array(times), np.array(cost), 'b')
plt.plot(np.array(times2), np.array(cost2), 'b--')


filename = 'SS_jw_50.npz'
data = np.load(filename)
times = data['T_ss']
cost = data['C_ss']

filename = 'SS_jw_50_sub.npz'
data = np.load(filename)
times2 = data['T_ss']
cost2 = data['C_ss']

"""
filename = 'jw.npz'
data = np.load(filename)
times3 = data['T_ddp']
cost3 = data['C_ddp']
print(cost)
print(cost3)

filename = 'M_jw_50.npz'
data = np.load(filename)
times3 = data['T_m']
cost3 = data['C_m']

total_data = np.concatenate((cost, cost2, cost3), axis=0)
mind = min(total_data)
maxd = max(total_data)
cost = (cost - mind)/(maxd-mind)
cost2 = (cost2 - mind)/(maxd-mind)
cost3 = (cost3 - mind)/(maxd-mind)

plt.plot(np.array(times), np.array(cost), 'm')
# plt.plot(np.array(times2), np.array(cost2), 'm--')
plt.plot(np.array(times3), np.array(cost3), 'r')
"""


# filename = 'jw.npz'
# data = np.load(filename)
# times = data['T_ddp']
# cost = data['C_ddp']
# times2 = data['T_ss']
# cost2 = data['C_ss']
# total_data = np.concatenate((cost,cost2),axis=0)
# mind = min(total_data)
# maxd = max(total_data)
# cost = (cost - mind)/(maxd-mind)
# cost2 = (cost2-mind)/(maxd-mind)

# plt.plot(np.array(times),np.array(cost),'m')
# plt.plot(np.array(times2),np.array(cost2),'m--')

filename = 'M_ctplt.npz'
data = np.load(filename)
times = data['T_m']
cost = data['C_m']

filename = 'ctplt.npz'
data = np.load(filename)
times2 = data['T_ss'].squeeze()
cost2 = data['C_ss'].squeeze()
total_data = np.concatenate((cost, cost2), axis=0)
mind = min(total_data)
maxd = max(total_data)
cost = (cost - mind)/(maxd-mind)
cost2 = (cost2-mind)/(maxd-mind)

plt.plot(np.array(times), np.array(cost), 'm')
plt.plot(np.array(times2), np.array(cost2), 'm--')

plt.ylabel('Loss')
plt.xlabel('Time, s')
plt.title('Comparing gradient-free to gradient-based methods')
#plt.legend(['CP_D', 'CP_S','DCP_D', 'DCP_S','JW_D', 'JW_S', 'C_S'])
plt.legend(['CP_D', 'CP_S', 'DCP_D', 'DCP_S', 'CTPLT_M', 'CTPLT_S'])  # ,'JW_D', 'JW_S', 'C_S'])
plt.show()
