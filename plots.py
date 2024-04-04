# %% Import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.colors as colors

mse = tf.keras.losses.MeanSquaredError()


# %% Colors

colors = np.array([
        'tab:blue', 'tab:orange', 'tab:green',
        'tab:red', 'tab:purple', 'tab:brown',
        'tab:pink', 'tab:gray', 'tab:olive'
        ])

colors33 = colors.reshape([3, 3])




def plot_b_sig(case, bs, sigs, sigs_m):
    
    ls = np.linspace(1, bs.shape[0], bs.shape[0])
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 4))
    
    axx = ax[0]
                        
    for i1 in range(6):
                            
        axx.plot(
                    ls,
                    sigs[:, i1],
                    linestyle='--',
                    marker='o',
                    markevery = 10,
                    label = f'{i1+1}',
                    color=colors[i1]
                    )
            
        axx.plot(
                    ls,
                    sigs_m[:, i1],
                    color=colors[i1]
                    )         


            
    axx.legend()
    axx.set_xlabel('$\\lambda$')
    axx.set_ylabel('$\sigma_{i}$')
    axx.set_title(case)    
    
    
    axx = ax[1]
                        
    for i1 in range(6):
                            
        axx.plot(
                    ls,
                    bs[:, i1],
                    linestyle='--',
                    marker='o',
                    markevery = 10,
                    label = f'{i1+1}',
                    color=colors[i1]
                    )


           
    axx.legend()
    axx.set_xlabel('$\\lambda$')
    axx.set_ylabel('$b_{i}$')



    plt.tight_layout()
    plt.show()