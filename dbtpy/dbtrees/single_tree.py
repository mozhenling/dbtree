
"""
Created on Mon Mar 15 19:42:56 2021

@author: mozhenling
"""
import time
from numpy.random import beta
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from tqdm import trange
import copy

class BanditsSolver:

    def __init__(self, solverType='ts', tradeoff = 1):
        self.solverType = solverType
        self.solver_thompson_sampling = ['ts1', 'ts', 'TS', 'Ts', 'tS',
                                         'Thompson samping', 'Thompson Samping',
                                         'thompson samping', 'thompson_samping']
        
        self.solver_ucb1 = ['ucb1','ucb','UCB1', 'UCB', 'Upper Confidence Bound 1',
                            'Upper Confidence Bound', 'upper confidence bound 1', 
                            'upper confidence bound']
        
        self.bandits_num = 0
        self.counts_bernuli = []
        self.values_bernuli = []
        self.values_para = []
        self.counts_para = []
        self.a = []
        self.b = []
        # balance exploration and exploitation in ucb1
        self.tradeoff  = tradeoff  

    def set_arms(self, n_arms):
        """
        set the arms of the parent, each arm corresponds to a child
        """
        self.bandits_num = n_arms
        self.counts_bernuli = [0 for col in range(n_arms)]
        self.values_bernuli = [0.0 for col in range(n_arms)]
        self.counts_para = [0 for col in range(n_arms)]
        self.values_para = [0.0 for col in range(n_arms)]
        self.a = [1 for arm in range(n_arms)]
        self.b = [1 for arm in range(n_arms)]

    def select_arm(self, variables=None, tradeoff = None, ucb_sign = 1):
        """
        select an arm to do the simulation or partion
        
        input:
            -variables: the decision variables, if not none, do the partion selection
                        if is none do the simulation so as to evaluate obj, for future use
            -tradoff: ucb tradeoff
            -ucb_sign: -1 -> minimization, 1-> maximaition
        output:
            -the selected arms
        """
        # us thompson smapling to select the arm
        if self.solverType in self.solver_thompson_sampling:
            # select an arm to do simulation
            if variables is None:
                return np.argmax([beta(self.a[i], self.b[i]) for i in range(self.bandits_num)])
            # select an variable to do partion, i.e., partion a dimention of the search space
            # else:
            #     return max(variables, key=lambda var: beta(variables[var]['var_arms'][0], variables[var]['var_arms'][1]) )
            
        # select arm based on UCB1
        if self.solverType in self.solver_ucb1:
            tradeoff = tradeoff  if tradeoff is not None else self.tradeoff 
            # Pick the best one with consideration of upper confidence bounds.
            # for tradoff = 1, the upper confidence bound is derived from the hoeffding's inequality (let p=t^-4), a.k.a, ucb1
            # if the node has not been visit before, it has the largest uncertainty #ucb_sign * self.values_para[indexes]
            UCB1 = np.inf * np.ones(self.bandits_num)
            for ind, visit_num in enumerate(self.counts_para):
                if visit_num != 0 :
                   UCB1[ind] = tradeoff  * np.sqrt( 2*np.log( sum(self.counts_para  ) ) / (  self.counts_para[ind]) )
                   
            return np.argmax([ucb_sign * value + ucb1 for value, ucb1 in zip(self.values_para, UCB1)])
            # return max(range(self.bandits_num), key=lambda indexes: ucb_sign * self.values_para[indexes] + tradeoff  * np.sqrt(
            #     2*np.log( sum(self.counts_para  ) + 1) / ( 1 +  self.counts_para[indexes]) ) ) 
                # + 1 means when selecting a child node, the visits of this node and the parent are already considered
                # at that time self.counts_para are zeros

    def update_arm(self, chosen_arm=None, reward_bernuli=None, reward_para=None):
        # if self.solverType in self.solver_thompson_sampling:
        self.counts_bernuli[chosen_arm] += 1
        self.values_bernuli[chosen_arm] = self.moving_avarage(self.values_bernuli[chosen_arm], reward_bernuli, self.counts_bernuli[chosen_arm])
        self.counts_para[chosen_arm] += 1
        self.values_para[chosen_arm] = self.moving_avarage(self.values_para[chosen_arm], reward_para, self.counts_para[chosen_arm])
        self.a[chosen_arm] += reward_bernuli
        self.b[chosen_arm] += 1 - reward_bernuli

    def moving_avarage(self, values_old, value_new, values_new_num):
        return (values_new_num - 1) / float(values_new_num) * values_old + 1 / float(values_new_num) * value_new


class TreeNode(BanditsSolver):

    def __init__(self, variables, nodeID, sovlerType, tradeoff):
        super(TreeNode, self).__init__(sovlerType, tradeoff)
        self.variables = variables
        self.var_names = list(variables.keys())
        self.id_split = 0
        self.nodeID = nodeID
        self.parentID = None
        self.childrenIDs = []
        self.children_search = False
        self.successful = False
        self.obj_value = 0

    def update_variables(self, var_name, paraMin, paraMax, Bernoulli_reward, change=1):
        """
        set the range of decision variables
        update the arms of choosing a dimension to be split
        """
        self.variables[var_name]=(paraMin, paraMax)
        # for future use
        # self.variables[var_name]['var_arms'][0] += Bernoulli_reward * change
        # self.variables[var_name]['var_arms'][1] += (1 - Bernoulli_reward) * change

    def add_child_ID(self, childID):
        self.childrenIDs.append(childID)

    def set_parent_ID(self, parentID):
        self.parentID = parentID

    def draw_variable(self):
        return {var:random.uniform(self.variables[var][0], self.variables[var][1]) for var in self.variables}


class DynamicBanditTree:

    def __init__(self, variables,  treeID=[], solverType='ts', tradeoff=1, treeType=3, nLevel_max=4):
        self.treeID = treeID
        self.search_level = 1
        self.treeType = treeType
        self.solverType = solverType
        self.nLevel_max = nLevel_max
        self.tradeoff = tradeoff
        self.parents_selected = []
        self.children_selected = []
        self.path_selected = []
        self.paraOpt = []
        self.banditOptID = []
        # we may use a different partion strategy in the future
        self.split = 'alter'# for now, we split each dimension one by one when grow the tree #split # or thomp
        self.var_try_list = []
        
        
        self.bandits = [TreeNode(variables=variables, nodeID=(1, 1, 0), sovlerType=solverType, tradeoff=tradeoff)]
        self.bandits[0].children_search = True
        self.dbtree_grow(self.bandits[0].nodeID, change=0)
        
    def dbtree_grow(self, parentID, treeType=None, Bernoulli_reward=1, change=1):
        treeType = self.treeType if treeType is None else treeType
        self.bandits[parentID[2]].set_arms(treeType)
        
        #-- select the split dimention alternatively
        if self.split ==  'alter':
            var_be_split = self.bandits[parentID[2]].var_names[self.bandits[parentID[2]].id_split]
        # elif self.split == 'thomp':
        #     var_be_split = self.bandits[parentID[2]].select_arm(self.bandits[parentID[2]].variables)
        
        p_paraMin = self.bandits[parentID[2]].variables[var_be_split][0]
        p_paraRes = self.bandits[parentID[2]].variables[var_be_split][1] - p_paraMin
        
        for i in range(treeType):
            depth_index, width_index, list_index = parentID[0] + 1, i, len(self.bandits)
            childID = (depth_index, width_index, list_index)
            self.bandits[parentID[2]].add_child_ID(childID)
            self.bandits.append(TreeNode(variables=copy.deepcopy(self.bandits[parentID[2]].variables), nodeID=childID,
                                         sovlerType=self.solverType, tradeoff= self.tradeoff))
            self.bandits[childID[2]].set_parent_ID(parentID)
            self.bandits[childID[2]].update_variables(var_name=var_be_split, paraMin=p_paraMin + i * p_paraRes / treeType,
                                                      paraMax=p_paraMin + (i + 1) * p_paraRes / treeType,
                                                      Bernoulli_reward=Bernoulli_reward,
                                                      change=change)
            if self.split == 'alter':
                self.bandits[childID[2]].id_split = self.bandits[parentID[2]].id_split + 1 if self.bandits[parentID[2]].id_split + 1 < len(self.bandits[parentID[2]].var_names) else 0

    def dbtree_pull_arms(self, tradeoff = None, ucb_sign = 1):
        self.parents_selected = []
        self.path_selected = []
        self.parents_selected.append(self.bandits[0])
        for j in range(self.nLevel_max - 2):
            if self.parents_selected[j].children_search:
                self.path_selected.append(self.parents_selected[j].select_arm(tradeoff  = tradeoff , ucb_sign = ucb_sign))
                self.parents_selected.append(self.bandits[self.parents_selected[j].childrenIDs[self.path_selected[j]][2]])
            else:
                break
        
        return self.parents_selected[-1].draw_variable()

    def dbtree_update(self, para_tried, Bernoulli_reward):
        simID = self.parents_selected[-1].nodeID
        para_reward = para_tried['objValue']
        for i in range(len(self.path_selected)):
            self.bandits[self.parents_selected[i].nodeID[2]].update_arm(self.path_selected[i], Bernoulli_reward, para_reward)

        self.bandits[simID[2]].obj_value = para_reward
        self.var_try_list.append(para_tried)
        if Bernoulli_reward == 1:
            self.paraOpt.append(para_tried)
            self.banditOptID.append(simID)
            self.bandits[simID[2]].successful = True
            treeType = self.treeType # future use
        else:
            treeType = self.treeType
            
        self.bandits[simID[2]].children_search = True
        self.dbtree_grow(simID, treeType, Bernoulli_reward)

    def dbtree_moving_avarage(self, values_old, value_new, values_new_num):
        return (values_new_num - 1) / float(values_new_num) * values_old + 1 / float(values_new_num) * value_new

    def dbtree_get_optiml(self):
        paraOpt_reward_best = min(self.paraOpt_reward)
        paraOpt_reward_best_id = self.paraOpt_reward.index(paraOpt_reward_best)
        return (
         self.paraOpt[paraOpt_reward_best_id], paraOpt_reward_best, self.paraOpt, self.paraOpt_reward)

    def dbtree_show_2Dshape(self):
        plt.figure()
        for b in self.bandits:
            xl, xu = b.variables['x0']
            yl, yu = b.variables['x1']
            plt.plot((xl, xu), (yl, yl), color='g')
            plt.plot((xu, xu), (yl, yu), color='g')
            plt.plot((xu, xl), (yu, yu), color='g')
            plt.plot((xl, xl), (yu, yl), color='g')

        for var_try in self.var_try_list:
            plt.plot((var_try['x0']), (var_try['x1']), marker='.', color='r')

        plt.title('Tree of ' + self.treeID)
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.show()

class BanditTreeSearch:

    def __init__(self, tree_depth = 10, tree_type =2):
        self.tree_type = tree_type
        self.tree_depth = tree_depth
        self.var_try_list = []         # save the decision variables and their objective values
        self.var_opt_list = []         # the fitness, i.e, the optimal ones
        self.dbt = []                  # the tree objects

    def solve(self, obj_fun, vars_and_bounds, opt_mode='min', solver_type='ts', tradeoff= 0.5, epoch_num=1000, **kwargs):
        """
        Inputs:
            -objFcn             # the objective function
            -vars_and_bounds    # the decision varibles and their bounds
            -solver_type        # the different solvers for multi-armed bandit problem, future use
            -tradeoff           # small-> exploitationg, large->exploration
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
        
        #-- create tree object
        dbt = DynamicBanditTree(variables=vars_and_bounds,  treeID='singleTree', 
                                 solverType=solver_type, tradeoff=tradeoff, treeType=tree_type, nLevel_max=tree_depth)
        
        
        #######################################################################
        ##------------------------ initialial run
        # if it is minization problem and ucb is used, set ucb_sign = -1
        var_try_list.append(dbt.dbtree_pull_arms( ucb_sign = -1)) 
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
            # if it is minization problem and ucb is used, set ucb_sign = -1
            var_try_list.append(dbt.dbtree_pull_arms( ucb_sign = -1))
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
                
            dbt.dbtree_update(var_try_list[-1], reward_berno)
            
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
            # print('The Optimal = ', var_opt_list[-1], '\n')
            plt.show()
            

