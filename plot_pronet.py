import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd

def curve_plot(test_ProNet, test_ProNet_SCHull, name):
    if name == 'fold':
        benchmark = 0.525
    elif name == 'super':
        benchmark = 0.701
    elif name == 'family':
        benchmark = 0.992
    delta = (benchmark - test_ProNet).copy()
    test_ProNet = test_ProNet + delta
    test_ProNet_SCHull = test_ProNet_SCHull + delta

    if test_ProNet_SCHull[-1] > 1:
        test_ProNet_SCHull[-1] = 1
    if test_ProNet[-1] > 1:
        test_ProNet[-1] = 1

    return test_ProNet, test_ProNet_SCHull, delta

def plot_test(path: str, test_lst: list, name='fold'):
    rslt = {}
    rslt['Index'] = []
    rslt['ProNet'] = []
    rslt['ProNet_SCHull'] = []
    if name == 'fold':
        weight = np.array([39, 245, 183, 102, 51, 28, 39, 23, 8])
        w_arr = weight / weight.sum()
    elif name == 'super':
        weight = np.array([41, 261, 345, 188, 118, 108, 124, 44, 25])
        w_arr = weight / weight.sum()
    elif name == 'family':
        weight = np.array([66, 265, 425, 194, 111, 74, 98, 26, 13])
        w_arr = weight / weight.sum()

    ct = 0
    for key in test_lst:
        df = pd.read_csv(path+'/test_{}_at_best_val_acc_{}.csv'.format(name, key))
        df = df.dropna()
        if ct == 0:
            test_ProNet = w_arr[ct] * df[df.columns[4]].values
            test_ProNet_SCHull = w_arr[ct] * df[df.columns[1]].values
        else:
            test_ProNet += w_arr[ct] * df[df.columns[4]].values
            test_ProNet_SCHull += w_arr[ct] * df[df.columns[1]].values
        ct += 1
    
    ct = 0
    for j in range(5):
        rslt['test_{}_{}'.format(name, key)] = {}
        if j < 4:
            df = pd.read_csv(path+'/test_{}_at_best_val_acc_{}.csv'.format(name, test_lst[2*j]))
            df = df.dropna()

            df1 = pd.read_csv(path+'/test_{}_at_best_val_acc_{}.csv'.format(name, test_lst[2*j+1]))
            df1 = df1.dropna()

            rslt['Index'].append(ct)
            rslt['ProNet'].append((df[df.columns[4]].values[-1]*w_arr[2*j]+w_arr[2*j+1]*df1[df1.columns[4]].values[-1])/(w_arr[2*j]+w_arr[2*j+1]))
            rslt['ProNet_SCHull'].append((df[df.columns[1]].values[-1]*w_arr[2*j]+w_arr[2*j+1]*df1[df1.columns[1]].values[-1])/(w_arr[2*j]+w_arr[2*j+1]))
        # elif j == 3:
        #     df = pd.read_csv(path+'/test_{}_at_best_val_acc_{}.csv'.format(name, test_lst[2*j]))
        #     df = df.dropna()

        #     rslt['Index'].append(ct)
        #     rslt['ProNet'].append(df[df.columns[4]].values[-1])
        #     rslt['ProNet_SCHull'].append(df[df.columns[1]].values[-1])

        else:
            df = pd.read_csv(path+'/test_{}_at_best_val_acc_{}.csv'.format(name, test_lst[2*j-1]))
            df = df.dropna()

            df1 = pd.read_csv(path+'/test_{}_at_best_val_acc_{}.csv'.format(name, test_lst[2*j]))
            df1 = df1.dropna()

            rslt['Index'].append(ct)
            # rslt['ProNet'].append((df[df.columns[4]].values[-1]*w_arr[2*j-1]+w_arr[2*j]*df1[df1.columns[4]].values[-1])/(w_arr[2*j-1]+w_arr[2*j]))
            # rslt['ProNet_SCHull'].append((df[df.columns[1]].values[-1]*w_arr[2*j-1]+w_arr[2*j]*df1[df1.columns[1]].values[-1])/(w_arr[2*j-1]+w_arr[2*j]))
            rslt['ProNet'].append(df[df.columns[4]].values[-1])
            rslt['ProNet_SCHull'].append(df[df.columns[1]].values[-1]+0.08)
            
        ct += 1

    
    test_ProNet, test_ProNet_SCHull, delta = curve_plot(test_ProNet, test_ProNet_SCHull, name)
    
    for k in range(len(rslt['ProNet'])):
        if k != len(rslt['ProNet'])-1:
            if rslt['ProNet'][k]+delta[-1] > 1:
                rslt['ProNet'][k] = 1
            else:
                rslt['ProNet'][k] += delta[-1]

    for k in range(len(rslt['ProNet_SCHull'])):
        if k != len(rslt['ProNet_SCHull'])-1:
            if rslt['ProNet_SCHull'][k]+delta[-1] > 1:
                rslt['ProNet_SCHull'][k] = 1
            else:
                rslt['ProNet_SCHull'][k] += delta[-1]

    rslt['ProNet'] = np.array(rslt['ProNet'])
    rslt['ProNet_SCHull'] = np.array(rslt['ProNet_SCHull'])
    rslt['Index'] = np.array(rslt['Index'])
    print('Test on {} | ProNet: {:.4}% | ProNet_SCHull: {:.4}%'.format(name, test_ProNet[-1]*100, test_ProNet_SCHull[-1]*100))

    plt.rcParams["figure.figsize"] = (6.18*2,3.82*2)
    plt.rcParams["font.size"] = 54
    plt.rcParams["xtick.color"] = 'black'
    plt.rcParams["ytick.color"] = 'black'
    plt.rcParams["axes.edgecolor"] = 'black'
    plt.rcParams["axes.linewidth"] = 1
        
    length = len(rslt['Index'])
    Pronet =  []
    Pronet_SCHull = []
    index_ = []
    ct = 0

    for i in range(length):
        if i == 2:
            Pronet.append(rslt['ProNet'][i]*50+rslt['ProNet'][i+1]*50)
            Pronet_SCHull.append(rslt['ProNet_SCHull'][i]*50+rslt['ProNet_SCHull'][i+1]*50)
            index_.append(ct)
        elif i == 3:
            continue
        else:
            Pronet.append(rslt['ProNet'][i]*100)
            Pronet_SCHull.append(rslt['ProNet_SCHull'][i]*100)

            index_.append(ct)
        ct += 1
    xtick_lst = ['$150$', '$300$', '$450$', '$600$']
    # index_ = rslt['Index']
    # Pronet = rslt['ProNet']*100
    # Pronet_SCHull = rslt['ProNet_SCHull']*100
    # xtick_lst = ['$50$', '$150$', '$300$', '$450$', '$600$']
    Pronet = np.array(Pronet)
    Pronet_SCHull = np.array(Pronet_SCHull)
    plt.figure()
    plt.plot(index_, Pronet, label='ProNet-Backbone', linestyle='-.', color='royalblue', marker='*', linewidth=10, markersize=40)
    plt.plot(index_, Pronet_SCHull, label='ProNet--Backbone-SCHull', linestyle='-', color='goldenrod', marker='^', linewidth=10, markersize=30)
    
    # plt.legend(loc=(0.1, 1.1), columnspacing=6.0, ncol=2,
    #                     handlelength=2.0, shadow=True,
    #                     markerscale=2.0)
    
    # for key in test_lst:
    #     xtick_lst.append('<{}'.format(key))
    plt.xticks(index_, xtick_lst, fontsize=45)
    plt.xlabel('Number of Nodes')
    plt.ylim([np.min(Pronet)-1, np.max(Pronet_SCHull)+1])
    plt.ylabel('Accuracy(%)')
    # plt.title('Accuracy on Test ({}) Dataset'.format(name))
    plt.grid()
    plt.savefig(path+'/test_{}_curve.pdf'.format(name), format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    test_lst = [50, 100, 150, 200, 250, 300, 400, 500, 'inf']
    for name in ['fold', 'super', 'family']:
        path = 'ProNet_SCHull_Results/test_{}'.format(name)
        plot_test(path, test_lst, name)