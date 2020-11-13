import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
import matplotlib as mlp
import matplotlib.lines as mlines
color_list = ['red', 'green', 'blue', 'magenta', 'orange', 'black', ]
# plt.style.use('seaborn-notebook')
mlp.rc('font',family='Times New Roman')

csfont = {'fontname':'Times New Roman'}
def plotData(list_bdrate, title = 'Untitle', xlabel = 'DecodingTime', ylabel= 'BD-rate', fontsize=8):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.figure.set_size_inches(15, 10)
    ax.scatter([x.time for x in list_bdrate], [y.bdrate for y in list_bdrate])
    for _, n in enumerate(list_bdrate):
        ax.annotate(n.name, (n.time, n.bdrate), fontsize=fontsize)
    ax.set_xlabel('DecodingTime', fontsize=15)
    ax.set_ylabel('BD-rate', fontsize=15)
    ax.set_title(title)
    ax.set_yticks(np.arange(0, max([y.bdrate for y in list_bdrate]), 0.5))
    ax.grid(False)
    plt.savefig(title + '.png', dpi=300)



def SplitDataFrameFromUniqueColumns(df, columns):
    _dic = {}
    _unique = df[columns].unique()
    for n in _unique:
        _dic[n] =(df[df[columns]==n])
    return _dic



def plotScatterData(fig, ax, dic_df, xvalue, yvalue ,title = 'Untitle',xlabel = None, ylabel=None, fontsize=8, extra_label=None, extra_text = None, fig_legend=False):
    texts = []
    legends = []
    if extra_label is not None:
        legends = extra_label
    if extra_text is not None:
        texts = extra_text
    others_legends = []

    if isinstance(dic_df, dict):
        for i, (name ,df) in enumerate(dic_df.items()):
            # gpus = df[df.GPU==True]
            # if len(gpus):
            #     ax.scatter(gpus[xvalue], gpus[yvalue], label='isGPU', marker='o', s=140, facecolor='none', alpha=1,edgecolors='red')
            # if name=='VTM1.0' or name=='VTM1.1':
            #     name += '(No ALF)'
            #     legends.append(ax.scatter(df[xvalue], df[yvalue], label=name, marker='+'))
            # else:
            legends.append(ax.scatter(df[xvalue], df[yvalue], label=name, facecolor = color_list[i]))
            # for i,n in df.iterrows():
            #     ax.annotate(n['Name'], (n[xvalue], n[yvalue]), fontsize= fontsize)
            for i,n in df.iterrows():
                t = ax.text(n[xvalue], n[yvalue], n['Name'], ha='center', va='center')
                texts.append(t)
                # texts.append(plt.text(n[xvalue], n[yvalue]/1000000,n['Name']))
    else:
        df = dic_df
        legends.append(ax.scatter(df[xvalue], df[yvalue]))
        for i,n in df.iterrows():
            texts.append(plt.text(n[xvalue], n[yvalue],n['Name']))
    # others_legends.append(ax.scatter([],[], label='W/O ALF', marker='+', color='black'))
    # others_legends.append(ax.scatter([], [], label='isGPU', marker='o', s=140, facecolor='none',edgecolors='red'))
    box = ax.get_position()
    # other_legend = ax.legend(handles=others_legends, fontsize=15, loc='lower left', bbox_to_anchor=(1,0))
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    if fig_legend:
        # fig.legend(fontsize=12, loc='center right')
        fig.legend(handles=legends, fontsize=10, loc='center right')
    # ax.add_artist(other_legend)

    if xlabel is None:
        ax.set_xlabel(xvalue, fontsize=15, **csfont)
    else:
        ax.set_xlabel(xlabel, fontsize=15, **csfont)
    if ylabel is None:
        ax.set_ylabel(yvalue, fontsize=15, **csfont)
    else:
         ax.set_ylabel(ylabel, fontsize=15, **csfont)
    # ax.set_title(title, fontsize=25)
    ax.grid(linestyle ='dotted')
    adjust_text(texts,ax=ax, **csfont)

    return legends





if __name__=='__main__':
    cnn_nt = namedtuple('cnn_nt', ['Name', 'flops', 'model_size', 'required_memory', 'task'])
    # list_prop = [cnn_nt('EDSR', 1442.92, 1369859, 2147483648, 'Super resolution'),
    #              cnn_nt('RDN', 23212.1, 22123395, 4563402752, 'Super resolution'),
    #              cnn_nt('RCAN', 13002.42, 12467227, 2147483648, 'Super resolution'),
    #              cnn_nt('SRCNN', 290.86, 69251, 2147483648, 'Super resolution'),
    #              cnn_nt('RDN_Denoising', 4270.24, 4068579, 2818572288, 'Denoising'),
    #              cnn_nt('DIDN', 17950.95, 191111722, 2147483648, 'Segmentation'),
    #              cnn_nt('DnCNN', 587.61, 559363 ,536870912, 'Denoising')
    # ]

    list_prop = [cnn_nt('EDSR', 1442.92, 1.37, 2.147483648, 'Super resolution'),
                 cnn_nt('RDN', 23212.1, 22.12, 4.563402752, 'Super resolution'),
                 cnn_nt('RCAN', 13002.42, 12.47, 2.147483648, 'Super resolution'),
                 cnn_nt('SRCNN', 290.86, 0.07, 2.147483648, 'Super resolution'),
                 cnn_nt('RDN_Denoising', 4270.24, 4.07, 2.818572288, 'Denoising'),
                 # cnn_nt('DIDN', 17950.95, 191.11, 2.147483648, 'Segmentation'),
                 cnn_nt('DnCNN', 587.61, 0.559363 , 0.536870912, 'Denoising'),
                 cnn_nt('FCNSS', 591.55, 136.15, 0.764, 'Segmentation')
    ]
    df = pd.DataFrame(data = list_prop)


    # inter_groupby_version = SplitDataFrameFromUniqueColumns(df[df['isIntra']==False], 'version')
    (fig_width, fig_height) = plt.rcParams['figure.figsize']
    fig_size = [fig_width*2.5, fig_height]
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=fig_size)

    grouped = SplitDataFrameFromUniqueColumns(df, 'task')

    plotScatterData(fig, ax[0], grouped, yvalue='model_size', ylabel='model size(MB)', xvalue='flops', xlabel='flops(GB)')
    plotScatterData(fig, ax[1], grouped, xvalue='flops', xlabel='flops(GB)', ylabel='required memory(GB)', yvalue='required_memory')
    l = plotScatterData(fig, ax[2], grouped, xvalue='model_size', xlabel='model size(MB)', ylabel='required memory(GB)', yvalue='required_memory', fig_legend=True)
    #
    # fig.legend([l1, l2, l3, l4],  # The line objects
    #            labels=line_labels,  # The labels for each line
    #            loc="center right",  # Position of legend
    #            borderaxespad=0.1,  # Small spacing around legend box
    #            title="Legend Title"  # Title for the legend
    #            )
    # plt.figtext(0.99, 0.01, 'Replaced : Replaced In-loop Filter\nOnline : Online Training', horizontalalignment='right')
    plt.savefig('Intra_CNN_LoopFilter.png', dpi=300)


    # ours_intra = lf_prop('Ours', '', 'Sejong', 4.23,0,0,0,0,134,71411,'DF-SAO-ALF-CNN', '64', True, True, 'VTM6.1')
    # ours_inter = lf_prop('Ours', '', 'Sejong', 1.44,0.87,0.71,0,0,134,84495,'DF-SAO-ALF-CNN', '64', False, True, 'VTM6.1')
    #
    #
    # namedBDrate = namedtuple('namedBDrate', ['name', 'bdrate', 'time'])
    # list_bdrate_intra = []
    # for n in list_prop:
    #     if not n.isIntra:
    #         continue
    #     # name = '('+n.title + ')(' + n.isReplaced_LoopFilter + ')'
    #
    #     name = '('+n.title + ')('+ n.company + ')(' + n.isReplaced_LoopFilter + ')'
    #     if not n.isOffline:
    #         name += '_Online'
    #     if n.dec_time_cpu is not None:
    #         list_bdrate_intra.append(namedBDrate(name, n.Y_gain, n.dec_time_cpu))
    #     else:
    #         name += '_gpu'
    #         list_bdrate_intra.append(namedBDrate(name, n.Y_gain, n.dec_time_gpu))
    #
    #
    # plotData(list_bdrate_intra, title='Bdrate-per-Intra', fontsize=8)
    # list_bdrate_inter = []
    # for n in list_prop:
    #     if n.isIntra:
    #         continue
    #     name = '('+n.title + ')(' + n.isReplaced_LoopFilter + ')'
    #
    #     # name = '('+n.title + ')('+ n.company + ')(' + n.isReplaced_LoopFilter + ')'
    #     if not n.isOffline:
    #         name += '_Online'
    #     if n.dec_time_cpu is not None:
    #         list_bdrate_inter.append(namedBDrate(name, n.Y_gain, n.dec_time_cpu))
    #     else:
    #         name += '_gpu'
    #         list_bdrate_inter.append(namedBDrate(name, n.Y_gain, n.dec_time_gpu))
    #     name += n.version
    # plotData(list_bdrate_inter, title='Bdrate-per-Inter', fontsize=8)


