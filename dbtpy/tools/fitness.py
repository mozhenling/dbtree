
import matplotlib.pyplot as plt

def plot_fitness(var_opt_list,findex='objValue',  figsize = (3.5, 1.8), dpi = 144,
             fig_save_path= None, fig_format = 'png', fontsize = 8):
    """
    plot the fitness
    """
    # if type(var_opt_list) is dict:
    var_opt = [var_list[findex] for var_list in var_opt_list]
    # elif type(var_opt_list) is list:
    #     var_opt = var_opt_list
    # else:
    #     raise ValueError
        
    indexes = range(len(var_opt))
    plt.figure(figsize = figsize, dpi = dpi)
    plt.rc('font', size = fontsize)
    
    plt.plot(indexes, var_opt)
    
    # plt.title('Trajectory of fittness', fontsize = fontsize + 1)
    plt.xlabel('Query',fontsize = fontsize + 0.5 )
    plt.ylabel('Fitness', fontsize = fontsize + 0.5)
    
    plt.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.show() 