from plot_helper import *



# Load and process data
log2_batch_size = lambda x: int(23 - np.ceil(np.log2(x)))
names = list([
    f'../data/ReLU_lr_sweep_no_bias_equal/autoencoder_ReLU_k_1024_batch_13_steps_17_learning_rate_{lr}_sample_equal_init_bias_0.0_decay_0.0_eps_0.015625_m_64_N_512_density_1.0_drpk_0.0.pt'
    for lr in [0.001,0.003,0.005,0.007,0.01,0.03]
])
ReLU_equal_lr_sweep = []
for n in names:
    try:
        ReLU_equal_lr_sweep.append(torch.load(n, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(n,'not found')

fig = training_plot(ReLU_equal_lr_sweep, 'learning_rate', log_color=True)
fig.tight_layout()
fig.savefig('../writeup/ReLU_equal_lr_sweep_training_plot_zero_bias.pdf', )

fig = sfa_plot(ReLU_equal_lr_sweep, 'learning_rate', [0,2,5])
fig.tight_layout()
fig.savefig('../writeup/ReLU_equal_lr_sweep_sfa_plot_zero_bias.pdf')

fig = plot_bias(ReLU_equal_lr_sweep, 'learning_rate', log_color=True)
fig.tight_layout()
fig.savefig('../writeup/ReLU_equal_lr_sweep_bias_plot_zero_bias.pdf')

# Load and process data
log2_batch_size = lambda x: int(23 - np.ceil(np.log2(x)))
names = list([
    f'../data/ReLU_lr_sweep_no_bias_power_law/autoencoder_ReLU_k_1024_batch_13_steps_17_learning_rate_{lr}_sample_power_law_init_bias_0.0_decay_0.0.pt'
    for lr in [0.001,0.003,0.005,0.007,0.01,0.03]
])
ReLU_power_law_lr_sweep = []
for n in names:
    try:
        ReLU_power_law_lr_sweep.append(torch.load(n, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(n,'not found')

fig = training_plot(ReLU_power_law_lr_sweep, 'learning_rate', log_color=True)
fig.tight_layout()
fig.savefig('../writeup/ReLU_power_law_lr_sweep_training_plot_zero_bias.pdf', )

fig = sfa_plot(ReLU_power_law_lr_sweep, 'learning_rate', [0,2,5])
fig.tight_layout()
fig.savefig('../diagnostic_plots/ReLU_power_law_lr_sweep_sfa_plot_zero_bias.pdf')

fig = plot_bias(ReLU_power_law_lr_sweep, 'learning_rate', log_color=True)
fig.tight_layout()
fig.savefig('../diagnostic_plots/ReLU_power_law_lr_sweep_bias_plot_zero_bias.pdf')

# Load and process data
log2_batch_size = lambda x: int(23 - np.ceil(np.log2(x)))
names = list([
    f'../data/ReLU_lr_sweep_negative_bias_equal/autoencoder_ReLU_k_1024_batch_13_steps_17_learning_rate_{lr}_sample_equal_init_bias_-1.0_decay_0.03_eps_0.015625_m_64_N_512_density_1.0_drpk_0.0.pt'
    for lr in [0.001,0.003,0.005,0.007,0.01,0.03]
])
ReLU_equal_lr_sweep = []
for n in names:
    try:
        ReLU_equal_lr_sweep.append(torch.load(n, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(n,'not found')

fig = training_plot(ReLU_equal_lr_sweep, 'learning_rate', log_color=True)
fig.tight_layout()
fig.savefig('../writeup/ReLU_equal_lr_sweep_training_plot_negative_bias.pdf', )

fig = sfa_plot(ReLU_equal_lr_sweep, 'learning_rate', [0,2,5])
fig.tight_layout()
fig.savefig('../writeup/ReLU_equal_lr_sweep_sfa_plot_negative_bias.pdf')

fig = plot_bias(ReLU_equal_lr_sweep, 'learning_rate', log_color=True)
fig.tight_layout()
fig.savefig('../writeup/ReLU_equal_lr_sweep_bias_plot_negative_bias.pdf')

# Load and process data
log2_batch_size = lambda x: int(23 - np.ceil(np.log2(x)))
names = list([
    f'../data/ReLU_lr_sweep_negative_bias_power_law/autoencoder_ReLU_k_1024_batch_13_steps_17_learning_rate_{lr}_sample_power_law_init_bias_-1.0_decay_0.03.pt'
    for lr in [0.001,0.003,0.005,0.007,0.01,0.03]
])
ReLU_power_law_lr_sweep = []
for n in names:
    try:
        ReLU_power_law_lr_sweep.append(torch.load(n, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(n,'not found')

fig = training_plot(ReLU_power_law_lr_sweep, 'learning_rate', log_color=True)
fig.tight_layout()
fig.savefig('../writeup/ReLU_power_law_lr_sweep_training_plot_negative_bias.pdf')
