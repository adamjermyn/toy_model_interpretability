from model_helper import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import matplotlib.cm as cmx

font = {
        'weight' : 'normal',
        'size'   : 18
        }
matplotlib.rc('font', **font)

nice_names = {
    'initial_bias': 'Initial Mean Bias',
    'learning_rate': 'Learning Rate',
    'k': '# of Neurons',
    'decay': 'Weight Decay Rate',
    'eps': r'$\epsilon$',
    'reg': 'L1 Regularization Strength'
}


def training_plot(batch, sweep_var, log_color=True, cm='Blues', loss_range=None):   
    sweep_vars = list(b[sweep_var] for b in batch)
    if log_color:
        sweep_vars_col = np.log10(sweep_vars)
    else:
        sweep_vars_col = sweep_vars

    sweep_ran_col = max(sweep_vars_col) - min(sweep_vars_col)
    
    ncol = 1 + int(len(batch) / 4)
    
    vmin = min(sweep_vars_col) - 0.3*sweep_ran_col
    vmax = max(sweep_vars_col) + 0.1*sweep_ran_col
    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0, width_ratios=[0.96,0.04], wspace=0)
    gs2 = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0, width_ratios=[0.97,0.03], wspace=0)

    fig = plt.figure(figsize=(13,9))
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[2,0])
    
    cax = fig.add_subplot(gs2[:,1])

    plot_data = []
    for j in range(len(batch)):
        plot_data.append([])
        for i in range(len(batch[j]['log2_spaced_models'])):
            sfa = single_feature_activations(batch[j]['log2_spaced_models'][i], batch[j], batch[j]['setup'])
            frac_mono = np.amax(sfa,axis=1) / (1e-10 + np.sum(sfa,axis=1))
            mean_bias = torch.mean(batch[j]['log2_spaced_models'][i]['0.bias']).numpy()
            plot_data[-1].append((i,sum(frac_mono > r_threshold), mean_bias))

    plot_data = np.array(plot_data)
    for j in range(len(batch)):
        col = scalarMap.to_rgba(sweep_vars_col[j])
        ax1.plot(np.log2(np.arange(1,2**4*len(batch[j]['losses'])+1,2**4)), batch[j]['losses'], label=f'{sweep_vars[j]}', c=col)
        ax2.plot(plot_data[j][:,0],plot_data[j][:,1]/512, c=col)
        ax3.plot(plot_data[j][:,0],plot_data[j][:,2], c=col)

    if loss_range is not None:
        ax1.set_ylim(loss_range)
        
    ax1.set_xticks([])
    ax2.set_xticks([])

    ax1.set_ylabel('Loss')
    ax2.set_ylabel('# Mono Neurons\n / # Features')
    ax3.set_ylabel('Mean bias')

    if log_color:
        clabel = r'$\log_{10}$'+f'{nice_names[sweep_var]}'
    else:
        clabel = str(nice_names[sweep_var])

    cb = fig.colorbar(scalarMap, cax=cax, orientation='vertical', label=clabel)


    ax2.axhline(1, linestyle=':', c='k')
    ax3.set_xlabel(r'$\log_2 \mathrm{Training\ Steps}$')
    
    ax1.set_title(f'Sweeping {nice_names[sweep_var]}')
    return fig

def plot_bias(batch, sweep_var, log_color=True, cm='Blues'):
    sweep_vars = list(b[sweep_var] for b in batch)
    if log_color:
        sweep_vars_col = np.log10(sweep_vars)
    else:
        sweep_vars_col = sweep_vars

    sweep_ran_col = max(sweep_vars_col) - min(sweep_vars_col)
    
    ncol = 1 + int(len(batch) / 4)
    
    cNorm  = colors.Normalize(vmin=min(sweep_vars_col) - 0.3*sweep_ran_col, vmax=max(sweep_vars_col) + 0.1*sweep_ran_col)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    fig = plt.figure(figsize=(13,9))

    plt.axhline(0, c='k', linestyle=':')

    for i in range(len(batch)):
        sfa = single_feature_activations(batch[i]['log2_spaced_models'][-1], batch[i], batch[i]['setup'])

        # Sort the neurons to put the most-monosemantic first
        inds = np.argsort(-np.amax(sfa,axis=1) / (1e-10 + np.mean(sfa,axis=1)))

        bias = batch[i]['log2_spaced_models'][-1]['0.bias'][inds]
        print(len(bias[bias > 0.05]))
        plt.plot(bias, label=str(sweep_vars[i]), c=scalarMap.to_rgba(sweep_vars_col[i]))
    plt.axvline(512,c='k', label='# Features')
    plt.xlabel('Neuron')
    plt.ylabel('Bias')

    max_k = max(batch[i]['k'] for i in range(len(batch)))
    plt.xlim([-0.1*max_k, 1.1*max_k])

    cax = fig.add_axes([0.11,0.9,0.25,0.02])
    cb = fig.colorbar(scalarMap, cax=cax, orientation='horizontal')
    if log_color:
        cax.set_title(r'$\log_{10}$' + nice_names[sweep_var])
    else:
        cax.set_title(f'{nice_names[sweep_var]}')

    return fig
    
def sfa_plot(batch, sweep_var, js, crop=[1024,512]):
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.04)

    fig = plt.figure(figsize=(13,18))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    axes = [ax1,ax2,ax3]

    for i,j in enumerate(js):
        sfa = single_feature_activations(batch[j]['log2_spaced_models'][-1], batch[j], batch[j]['setup'])

        # Sort the neurons to put the most-monosemantic first
        inds = np.argsort(-np.amax(sfa,axis=1) / (1e-10 + np.mean(sfa,axis=1)))
        sfa = sfa[inds]

        # Sort the features to put the most-monosemantic neurons first
        neuron_inds = []
        for k in range(sfa.shape[1]): # Loop over features
            neuron_ind = np.argmax(sfa[:,k]) # Find the neuron this feature activates most-strongly.
            neuron_inds.append(neuron_ind)
        inds = np.argsort(neuron_inds) # Sort the neuron indices
        sfa = sfa[:,inds]

        im = axes[i].imshow(sfa[:crop[0],:crop[1]],interpolation='nearest', aspect=1.8, vmin=0, vmax=1.02)
        axes[i].annotate(f'{nice_names[sweep_var]}={round(batch[j][sweep_var],4)}', (10,20), c='white')
        axes[i].set_xlabel('Feature')
    
    cbar = fig.colorbar(im, orientation='horizontal', ax=axes, location='top', pad=0.01, aspect=40)

    ax1.set_ylabel('Neuron')
    ax2.set_yticks([])
    ax3.set_yticks([])
    return fig

def mfa_plot(batch, sweep_var, js, extras):
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.06)

    fig = plt.figure(figsize=(15,20))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    axes = [ax1,ax2,ax3]

    for i,j in enumerate(js):
        sfa = many_feature_activations(batch[j]['log2_spaced_models'][-1], batch[j], batch[j]['setup'], extras)

        # Sort the neurons to put the most-monosemantic first
        inds = np.argsort(-np.amax(sfa,axis=1) / (1e-10 + np.mean(sfa,axis=1)))
        sfa = sfa[inds]

        im = axes[i].imshow(sfa.T,interpolation='nearest')
        axes[i].annotate(f'{nice_names[sweep_var]}={batch[j][sweep_var]}', (40,40), c='white')
        axes[i].set_ylabel('Feature')
        fig.colorbar(im, orientation='vertical', ax=axes[i])

    ax3.set_xlabel('Neuron')
    return fig

def sfa_line_plot(data, extras):
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0)

    fig = plt.figure(figsize=(7,6))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    axes = [ax1,ax2,ax3]
    
    sfa = single_feature_activations(data['log2_spaced_models'][-1], data, data['setup'])

    # Sort the neurons to put the most-monosemantic first
    inds = np.argsort(-np.amax(sfa,axis=1) / (1e-10 + np.mean(sfa,axis=1)))
    sfa = sfa[inds]

    for i in range(3):
        ax = axes[i]
        ax.plot(sfa[:,extras[i]])
        ax.set_ylabel(f'Activation\n Feature {extras[i]}')

    ax3.set_xlabel('Neuron')
    return fig

def plot_mono_sweep(batch, sweep_var):
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0)

    fig = plt.figure(figsize=(10,9))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    axes = [ax1,ax2]

    plot_data = []
    for j in range(len(batch)):
        sfa = single_feature_activations(batch[j]['log2_spaced_models'][-1], batch[j], batch[j]['setup'])
        frac_mono = np.amax(sfa,axis=1) / (1e-10 + np.sum(sfa,axis=1))
        plot_data.append((batch[j][sweep_var],sum(frac_mono > r_threshold), sum(frac_mono > r_threshold)/len(frac_mono)))

    plot_data = np.array(plot_data)

    ax1.plot(plot_data[:,0],plot_data[:,2])
    if sweep_var == 'k':
        ax1.axvline(512, linestyle=':', c='r')
    ax1.axhline(1, linestyle=':', c='k')
    ax1.set_ylabel('# Mono Neurons /\n # Neurons')
    ax1.set_xticks([])
        
    ax2.plot(plot_data[:,0],plot_data[:,1]/512)
    if sweep_var == 'k':
        plt.axvline(512, linestyle=':', c='r')
    ax2.axhline(1, linestyle=':', c='k')
    ax2.set_ylabel('# Mono Neurons /\n # Features')
    ax2.set_xlabel(nice_names[sweep_var])

    return fig

