#!/usr/bin/env python
# coding: utf-8

# # load useful libraries

# In[1]:


import scanpy as sc
import numpy as np
import pandas as pd

from warnings import filterwarnings
filterwarnings('ignore')

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')


# # read data

# In[3]:


adata = sc.read_h5ad('./sc_training.h5ad')
df_full = pd.DataFrame(adata.X.toarray())
df_full.columns = list(adata.var_names)
print(f"There are totally {len(adata.obs.state.unique())} states, they are: {list(adata.obs.state.unique())}.")
print(f"There are totally {len(adata.obs.condition.unique())} type of gene experiments, which is consistent to description that 66 gene knockouts with unperturb.")


# In[4]:


df_obs = adata.obs.copy()
df_obs = df_obs.reset_index()


# In[6]:


# 1) why there are only 64 rows in above table? 
experiment_genes = list(df_obs.condition.unique())
expression_genes = list(df_full.columns)
print("Following knockout genes are not features of gene expression matrix:")
for i in experiment_genes:
    if i not in expression_genes and i != "Unperturbed":
        print(i)


# In[10]:


known_experiment_genes = list(set(list(df_obs.condition.unique())).difference({"Unperturbed", "Fzd1", "P2rx7"}))

def simulate_state(knockout, samplesize=5000, Knn = 3, echos = 10):
    if knockout in known_experiment_genes:
        state_count = np.array(df_obs[df_obs.condition == knockout].groupby("state").state.count())
        state_ratio = state_count/sum(state_count)
        state_name = ["cycling", "effector", "other", "progenitor", "terminal exhausted"]
        state_map = {}
    
        for i in range(5):
            state_map[state_name[i]] = state_ratio[i]
        
        return state_map
        
    # sampling cells from dataset:
    df_sample = df_full.sample(n = samplesize).transpose()
    distance = []
    # getting 3 nearest neighbours:
    for known in known_experiment_genes:
        dist = sum(sum(np.square(np.array((df_sample[df_sample.index == known].values - df_sample[df_sample.index == knockout].values)))))
        distance.append(dist)

    df_dist = pd.DataFrame({"gene": known_experiment_genes,
                            "distance": distance})

    nn_genes = df_dist.sort_values(by = ["distance"]).reset_index()
    
    total_weight = 0 
    state_ratio = 0
    var = nn_genes.distance.var()+10
    
    for e in range(echos):
        k = 0
        i = 0 
    
        while k < Knn and i < len(nn_genes):
            cur = nn_genes.gene[i]
            if cur in known_experiment_genes and cur != knockout:
            
                k += 1
        
                weight = np.exp(-nn_genes.distance[i]/(2*var))
                total_weight += weight
            
                state_count = np.array(df_obs[df_obs.condition == cur].groupby("state").state.count())
                state_ratio += state_count*weight/sum(state_count)
            i = i+1
        
        
    
    state_ratio = state_ratio/total_weight
    state_name = ["cycling", "effector", "other", "progenitor", "terminal exhausted"]
    state_map = {}
    
    for i in range(5):
        state_map[state_name[i]] = state_ratio[i]
        
    return state_map


# In[26]:


all_genes = list(set(list(df_full.columns)))

def simulation_state_all_genes(all_genes,echos = 10, samplesize=5000):

    simulation = np.zeros((len(all_genes),5))
    for e in range(echos):
        test_map = {}
        for t in all_genes:
            tmap = simulate_state(knockout = t, samplesize=samplesize, Knn = 3)
            formal_order = ['progenitor','effector','terminal exhausted','cycling','other']
            tlist = []
            for i in range(5):
                tlist.append(tmap[formal_order[i]])
            test_map[t] = tlist 
        
        test_output = pd.DataFrame(test_map).transpose()
        simulation = simulation+test_output.values
        
    simulation = pd.DataFrame(simulation/echos)
    simulation.columns = formal_order
    simulation.index = all_genes
    return simulation


# In[ ]:


all_genes = list(df_full.columns)

## !!!!!! Please try "echos" as larger as possible to stablize the results!!!!!!!
simulations = simulation_state_all_genes(all_genes,echos = 3, samplesize = 5000)


# In[ ]:


# simulations.to_csv("simulations_3_5000.csv")


# In[ ]:


#import pandas as pd
#simulations = pd.read_csv("simulations_i.csv")
#simulations = simulations.rename({"Unnamed: 0": "gene"}, axis = 1)


# In[ ]:


simulations = simulations.reset_index().rename({"index":"gene"}, axis = 1)


# In[15]:


## part a 


# In[ ]:


simulations_A = simulations.sort_values(by = "progenitor", ascending = False)
simulations_A["constraint"] = simulations_A.apply(lambda x: 1 if x.cycling >=0.05 else 0, axis = 1)
part_a_output = simulations_A[["gene","progenitor", "constraint"]]
part_a_output = part_a_output.set_index("gene")
part_a_output.to_csv("part_a_output.csv")


# In[ ]:





# In[17]:


## part b


# In[18]:


simulations["objective function"] = simulations.progenitor/0.0675 + simulations.effector/0.2097 - simulations["terminal exhausted"]/0.3134 + simulations.cycling/0.3921
simulations_B = simulations.sort_values(by = "objective function", ascending = False)
simulations_B["constraint"] = simulations_B.apply(lambda x: 1 if x.cycling >=0.05 else 0, axis = 1)
part_b_output = simulations_B[["gene","objective function", "constraint"]]
part_b_output = part_b_output.set_index("gene")
part_b_output.to_csv("part_b_output.csv")


# In[ ]:





# In[19]:


## part c


# In[21]:


#all_genes = list(set(list(df_full.columns)))

## !!!!!! Please try "echos" as larger as possible to stablize the results!!!!!!!
simulations = simulation_state_all_genes(all_genes,echos = 5, samplesize = 5000)
part_c_output = simulations


# In[ ]:


part_c_output = simulations.set_index("gene")
part_c_output.to_csv("part_c_output.csv")

