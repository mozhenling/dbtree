
"""
Created on Mon Mar 15 19:42:56 2021

@author: mozhenling
"""
import numpy as np
from numpy import random
from numpy.random import beta
import matplotlib.pyplot as plt
from tqdm import trange
import copy

#%%--------------------- Maintaining the tree node selections
class BanditsSolver:
    def __init__(self, solver_type='ts', explo_raORit_c=0.5):
        self.solver_type = solver_type
        self.solver_thompson_sampling = ['ts3','ts2', 'rts', 'ts']
        self.bandits_num = 0
        self.a = []
        self.b = []
        self.explo_raORit_c = explo_raORit_c

    def set_arms(self, n_arms):
        """
        set the arms for the parent, each arms corresponds to a child
        """
        self.bandits_num = n_arms
        self.a = [1 for arm in range(n_arms)]
        self.b = [1 for arm in range(n_arms)]

    def select_arm(self):
        if self.solver_type in self.solver_thompson_sampling:
            return np.argmax([beta(self.a[i], self.b[i]) for i in range(self.bandits_num)])

    def update(self, chosen_arm=None, reward_bernuli=None, reward_para=None, para_tried=None, beta_update_info=1.0):
        if self.solver_type in self.solver_thompson_sampling:
            #-- update the beta posterior
            # if the solver is ts_normal, we do not reshape the posterior
            beta_update_info = beta_update_info if self.solver_type in ['ts3', 'rts'] else 1
            self.a[chosen_arm] += reward_bernuli * beta_update_info
            self.b[chosen_arm] += (1 - reward_bernuli) * beta_update_info

    def moving_avarage(self, values_old, value_new, values_new_num):
        """incremental average"""
        return (values_new_num - 1) / float(values_new_num) * values_old + 1 / float(values_new_num) * value_new

#%%--------------------- Maintaining the tree nodes
class TreeNode(BanditsSolver):
    def __init__(self, nodeID, sovler_type, explo_raORit_c):
        super(TreeNode, self).__init__(sovler_type, explo_raORit_c)
        self.nodeID = nodeID
        self.parentID = None
        self.childrenIDs = []
        self.children_search = False
        self.successful = False
        self.obj_value = 0

    def set_range(self, paraMin=0.0, paraMax=1000.0):
        """
        set the range of decision variables
        """
        self.paraMin = paraMin
        self.paraMax = paraMax
        self.paraResolution = self.paraMax - self.paraMin

    def add_child_ID(self, childID):
        self.childrenIDs.append(childID)

    def set_parent_ID(self, parentID):
        self.parentID = parentID

    def draw_variable(self):
        return random.uniform(self.paraMin, self.paraMax)

#%%--------------------- Maintaining the DBtree
class DynamicBanditTree:
    def __init__(self, treeID=[], solver_type='ts', explo_raORit_c=2, objType='Min', treeType=2, nLevel_max=4):
        self.treeID = treeID
        self.ids = []
        self.search_level = 1
        self.treeType = treeType
        self.objType = objType
        self.solverType = solver_type
        self.explo_raORit_c = explo_raORit_c
        self.nLevel_max = nLevel_max
        self.bandits = [ TreeNode(nodeID=(1, 1, 0), sovler_type=solver_type, explo_raORit_c=explo_raORit_c) ]
        self.parents_selected = []
        self.children_selected = []
        self.path_selected = []
        self.paraOpt = []
        self.paraOpt_reward = []
        self.banditOptID = []

    def dbtree_grow(self, parentID, treeType=None):
        treeType = self.treeType if treeType is None else treeType
        self.bandits[parentID[2]].set_arms(treeType)  # create arms for each child
        p_paraMin = self.bandits[parentID[2]].paraMin
        p_paraRes = self.bandits[parentID[2]].paraResolution
        
        #-- create children
        for i in range(treeType):
            #-- create child id
            depth_index, width_index, list_index = parentID[0] + 1, i, len(self.bandits)
            childID = ( depth_index, width_index, list_index )
            
            #-- create child node and set relationship
            self.bandits[parentID[2]].add_child_ID(childID)
            self.bandits.append( TreeNode( nodeID=childID, sovler_type=self.solverType, explo_raORit_c=self.explo_raORit_c )   )
            self.bandits[childID[2]].set_parent_ID(parentID)
            self.bandits[childID[2]].set_range(paraMin = p_paraMin + i * p_paraRes / treeType,
                                               paraMax = p_paraMin + (i + 1) * p_paraRes / treeType )

    def dbtree_set(self, treeType=2, nLevel_max=4, paraMin=0.0, paraMax=1000.0):
        """initialize a tree and create the root and its first children"""
        self.treeType = treeType
        self.nLevel_max = nLevel_max
        self.bandits[0].set_arms(n_arms=(self.treeType))
        self.bandits[0].set_range(paraMin=paraMin, paraMax=paraMax)
        self.bandits[0].children_search = True
        self.dbtree_grow(self.bandits[0].nodeID)

    def dbtree_pull_arms(self):
        self.parents_selected = []
        self.path_selected = []
        self.parents_selected.append(self.bandits[0])
        
        #-- select a branch
        for j in range(self.nLevel_max - 2):
            if self.parents_selected[j].children_search:
                self.path_selected.append(self.parents_selected[j].select_arm())
                self.parents_selected.append(self.bandits[self.parents_selected[j].childrenIDs[self.path_selected[j]][2]])
            else:
                break
            
        #-- draw random variables
        return self.parents_selected[-1].draw_variable()

    def dbtree_update(self, para_tried, Bernoulli_reward, para_reward, beta_update_info):
        # the node that does the simulation
        simID = self.parents_selected[-1].nodeID 
        #-- update parent nodes
        for i in range(len(self.path_selected)):
            self.bandits[self.parents_selected[i].nodeID[2]].update((self.path_selected[i]), Bernoulli_reward,
              para_reward, beta_update_info=beta_update_info)
            
        #-- record info.
        self.bandits[simID[2]].obj_value = para_reward
        if Bernoulli_reward == 1:
            self.paraOpt.append(para_tried)
            self.paraOpt_reward.append(para_reward)
            self.banditOptID.append(simID)
            self.bandits[simID[2]].successful = True
            
        #-- grow tree nodes
        self.bandits[simID[2]].children_search = True
        self.dbtree_grow(simID)

    def dbtree_moving_avarage(self, values_old, value_new, values_new_num):
        return (values_new_num - 1) / float(values_new_num) * values_old + 1 / float(values_new_num) * value_new

    def dbtree_get_optiml(self):
        paraOpt_reward_best = min(self.paraOpt_reward)
        paraOpt_reward_best_id = self.paraOpt_reward.index(paraOpt_reward_best)
        return (
         self.paraOpt[paraOpt_reward_best_id], paraOpt_reward_best, self.paraOpt, self.paraOpt_reward)

    def dbtree_show_history(self, iDim = 1, saveMode = False, plotMode = True):
        #-- plot the successful history of the Bernoulli bandit model
        #-- Regarding the meaning of "successful", please refer to Algoirthm I of the paper
        indexes = range(len(self.paraOpt_reward))
        if indexes == []:
            raise Exception('No optimal parameter is found after the first iteration of the simulation!')
        paraOpt_best, paraOpt_reward_best, _, _ = self.dbtree_get_optiml()
        #-- show the history of the reward (the successful obj. values)
        # sObjValue = []
        # sVariable = []
        if iDim==0:
            if plotMode:
                plt.figure()
                plt.plot(indexes, self.paraOpt_reward)
                plt.title('Success. record of the obj. value. The best: {:.2f}'.format(paraOpt_reward_best))
                plt.xlabel('Indexes')
                plt.ylabel('Reward')
                plt.show()
        #-- show the variables being tried under the successful trials
        if plotMode:
            plt.figure()
            plt.plot(indexes, self.paraOpt)
            plt.title('Success. record of ' + self.treeID + '. The best: {:.2f}'.format(paraOpt_best))
            plt.xlabel('Indexes')
            plt.ylabel('Variable')
            plt.show()
        if saveMode:
            sObjValue = [r for r in self.paraOpt_reward]
            sVariable = [v for v in self.paraOpt]
            return {'sObjValue':sObjValue, 'sVariable':sVariable}

    def dbtree_show_shape(self, saveMode = False, plotMode = True):
        """
        plot the structure of a tree, two dimensional space 
        """
        if saveMode:
            branches_list = [] 
        if plotMode:
            plt.figure()
        b = self.bandits
        #-- plot figures
        if plotMode:
            for i in trange(len(b)):
                if b[i].childrenIDs:
                    px, py = (b[i].paraMax + b[i].paraMin) / 2, b[i].nodeID[0]
                for bbID in b[i].childrenIDs:
                    cx, cy = (b[bbID[2]].paraMax + b[bbID[2]].paraMin) / 2, bbID[0]
                    X, Y = (px, cx), (py, cy)
                    if b[bbID[2]].successful:
                        plt.plot(X, Y, color='orange')
                    else:
                        plt.plot(X, Y, color='green')
            plt.gca().invert_yaxis()
            plt.title('Tree of ' + self.treeID)
            plt.xlabel('Tree width index')
            plt.ylabel('Tree depth index')
            plt.show()
        #-- save figure information   
        if saveMode:
            for i in range(len(b)):
                if b[i].childrenIDs:
                    px, py = (b[i].paraMax + b[i].paraMin) / 2, b[i].nodeID[0]
                for bbID in b[i].childrenIDs:
                    cx, cy = (b[bbID[2]].paraMax + b[bbID[2]].paraMin) / 2, bbID[0]
                    X, Y = (px, cx), (py, cy)
                    if b[bbID[2]].successful:
                        branches_list.append([X, Y, 'orange', 'win'])
                    else:
                        branches_list.append([X, Y, 'green', 'fail'])
            return branches_list
            
#%%--------------------- Search algorithm
class BanditTreeSearch:
    def __init__(self, tree_depth=10, tree_type=2):
        self.tree_type = tree_type
        self.tree_depth = tree_depth
        self.var_try_list = []         # save the decision variables and their objective values
        self.var_opt_list = []         # the fitness, i.e, the optimal ones
        self.dbt = []                  # the tree objects

    def reshape_posterior(self, vars_and_bounds, threshold, var_try_latest, var_try_opt_latest):
        """render non-uniform convergence of different decision variables"""
        var_change = [abs(var_try_opt_latest[var] - var_try_latest[var]) / (varMax - varMin) 
                       for var, (varMin, varMax) in vars_and_bounds.items()]
        
        dim = len(vars_and_bounds) # number of dimensions

        return [dim * var_c / (sum(var_change) + 1e-16) for var_c in var_change]

    def solve(self, obj_fun, vars_and_bounds, opt_mode='min', solver_type='ts', tradeoff=0.5, epoch_num=1000, **kwargs):
        """
        Inputs:
            -objFcn             # the objective function
            -vars_and_bounds    # the decision varibles and their bounds
            -solver_type        # the different solvers for multi-armed bandit problem, future use
            -tradeoff           # the forgetting rate, small-> exploitationg, large->exploration
            -epoch_num          # the number of epoches
            -**kwargs           # other keywords arguments of the objective function

        Outputs:
            self.var_try_list = var_try_list  # save the decision variables and their objective values
            self.var_opt_list = var_opt_list  # the fitness, i.e, the optimal variables and their objective values
            self.dbt = dbt                    # the tree objects
            print('The Optimal = ', var_opt_list[-1], '\n')

        """
        #-- initialize
        tree_type = self.tree_type
        tree_depth = self.tree_depth
        var_try_list = [] # save all variables and the objective values
        var_opt_list = [] # save all optimal variables and the objective values
        ndim_from_one_var=False
        
        #-- set n-dim variables from a single var input
        try:
            dim_num = vars_and_bounds['dim']
            ndim_keys = list(vars_and_bounds.keys()) 
            vars_and_bounds_old, vars_and_bounds = copy.deepcopy(vars_and_bounds), {}
            try:
                # different bounds for different dimensions, inputs:# 'vars_and_bounds':{'x':[(lb1,lb2),(ub1,ub2) ], 'dim':10}
                vars_and_bounds = {ndim_keys[0] + str(i):(vars_and_bounds_old[ndim_keys[0]][0][i],
                                                              vars_and_bounds_old[ndim_keys[0]][1][i])
                                                                               for i in range(dim_num)}
            except:
                # same bounds for all dimensions, input: # 'vars_and_bounds':{'x':(lb,ub), 'dim':10}
                vars_and_bounds = {ndim_keys[0] + str(i):vars_and_bounds_old[ndim_keys[0]] for i in range(dim_num)}
            
            # set flag of using one var to create ndim vars as true
            ndim_from_one_var = True
        except:
            pass
        
        #-- create trees 
        dbt = [DynamicBanditTree(var_name, solver_type, tradeoff) for var_name in vars_and_bounds]
        for i_tree, (_, (paraMin, paraMax)) in enumerate(vars_and_bounds.items()):
            dbt[i_tree].dbtree_set(tree_type, tree_depth, paraMin=paraMin, paraMax=paraMax)
            
        #######################################################################
        ##------------------------ initialial run
        #-- get actions and try the objective
        var_try_list.append({var_name:dbt[i_tree].dbtree_pull_arms() for i_tree, var_name in enumerate(vars_and_bounds)})
        var_try_list[-1]['objValue'] = obj_fun(**var_try_list[-1], 
                                                **kwargs) if ndim_from_one_var is False else obj_fun(
                                                    **{ndim_keys[0]:list(var_try_list[-1].values())}, **kwargs)
        
        #-- initialize
        threshold = var_try_list[-1]['objValue']
        var_opt_list.append(var_try_list[-1])
        var_opt_temp = var_try_list[-1]
        objValue_min_t = var_opt_temp['objValue']
        # print('initial run is included', '\n')
        
        #######################################################################
        ##------------------------ main run
        for i in trange(epoch_num - 1):
            #-- get actions and try the objective 
            var_try_list.append({var_name:dbt[i_tree].dbtree_pull_arms() for i_tree, var_name in enumerate(vars_and_bounds)})
            var_try_list[-1]['objValue'] = obj_fun(**var_try_list[-1], 
                                                **kwargs) if ndim_from_one_var is False else obj_fun(
                                                    **{ndim_keys[0]:list(var_try_list[-1].values())}, **kwargs)
                                                    
            objValue = var_try_list[-1]['objValue']
            #------------ Min or Max ? 
            if opt_mode in ['min', 'Min']:
                #-- Bernoulli reward
                if objValue < threshold:
                    reward_berno = 1
                    threshold = (1 - tradeoff) * objValue +  tradeoff * threshold 
                else:
                    reward_berno = 0
            elif opt_mode in ['max', 'Max']:
                #-- Bernoulli reward
                if objValue > threshold:
                    reward_berno = 1
                    threshold = (1 - tradeoff) * objValue +  tradeoff * threshold 
                else:
                    reward_berno = 0
            else:
                raise Exception('min or max ?')
            #-------------   
                    
            beta_update_info = self.reshape_posterior(vars_and_bounds, threshold, var_try_list[-1], var_opt_list[-1])
            
            #-- update trees
            for i_tree, var_name in enumerate(vars_and_bounds):
                dbt[i_tree].dbtree_update(var_try_list[-1][var_name], reward_berno, objValue, beta_update_info[i_tree])
            
            #------------ Min or Max ? 
            #-- record optimal
            if objValue < objValue_min_t and opt_mode in ['min', 'Min']:
                objValue_min_t = objValue
                var_opt_temp = var_try_list[-1]
                
            if objValue > objValue_min_t and opt_mode in ['max', 'Max']:
                objValue_min_t = objValue
                var_opt_temp = var_try_list[-1]
          
            var_opt_list.append(var_opt_temp)
                
        #######################################################################
        ##------------------------ save results
        self.var_try_list = var_try_list
        self.var_opt_list = var_opt_list
        self.dbt = dbt
        print('The Optimal = ', var_opt_list[-1], '\n')
          
    def tree_shape(self):
        #-- save the tree shape into a dictionary
        shapes_dict = {}
        for d in self.dbt:
            shapes_dict[d.treeID] = d.dbtree_show_shape(saveMode=True, plotMode = False)
        return shapes_dict
    def success_info(self):
        sInfo_dict = {}
        for d in self.dbt:
            sInfo_dict[d.treeID] = d.dbtree_show_history(iDim=d.treeID, saveMode=True, plotMode = False)
        return sInfo_dict
            
    def visualization(self, sim_update_show=False, tree_update_show=False, tree_shape_show=False):
        plt.close('all')
        #-- varaibles being tried and the corresponding objective values
        if tree_update_show:
            for i, d in enumerate(self.dbt):
                d.dbtree_show_history(i)
        #-- tree shape
        if tree_shape_show:
            for d in self.dbt:
                d.dbtree_show_shape()
        #-- optimization fitness
        if sim_update_show:
            opt_objValues = [var_list['objValue'] for var_list in self.var_opt_list]     
            indexes = range(len(opt_objValues))
            plt.figure()
            plt.plot(indexes, opt_objValues)
            plt.title('Trajectory of fitness and the best is: {:.2f}'.format(opt_objValues[-1]))
            plt.xlabel('Query')
            plt.ylabel('Fitness')
            # print('The Optimal = ', var_opt_list[-1], '\n')
            plt.show()
            
###############################################################################
#------------------------------------------------------------------------------
#-- customized plots
def plot_shape(branches_list, treeID, figsize = (3.5, 1.8), dpi = 144, s_color = None, f_color = None,
             fig_save_path= None, fig_format = 'png', fontsize = 8, non_text = False):
    
    """
    inputs:
        -branches_list = [x, y, color]
        -treeID, variable name
    """
    
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rc('font', size = fontsize)
    #-- 'orange': successful trial; 'green': unsuccessful trial
    for b in branches_list:
        b[2] = s_color if b[2] == 'orange' and s_color is not None else b[2]
        b[2] = f_color if b[2] == 'green' and f_color is not None else b[2]
 
        plt.plot(b[0], b[1],  b[2])
    
    plt.gca().invert_yaxis()
    
    if non_text: 
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
    else:
        plt.title('Tree of ' + treeID, fontsize = fontsize + 1)
        plt.xlabel('Frequency (Hz)',fontsize = fontsize + 0.5 )
        plt.ylabel('Depth', fontsize = fontsize + 0.5)
    
    plt.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.show()

def plot_fitness(var_opt_list, figsize = (3.5, 1.8), dpi = 144,
             fig_save_path= None, fig_format = 'png', fontsize = 8):
    """
    plot the fitness
    """
    if type(var_opt_list) is dict:
        var_opt = [var_list['objValue'] for var_list in var_opt_list]
    elif type(var_opt_list) is list:
        var_opt = var_opt_list
    else:
        raise ValueError
        
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