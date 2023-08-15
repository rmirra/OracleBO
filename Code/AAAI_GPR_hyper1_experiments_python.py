# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 22:49:30 2023

@author: rmirr
"""

import numpy as np
import math
from ax.utils.measurement.synthetic_functions import branin
import os
from scipy.io import wavfile
from librosa import stft, istft, load, mel_frequencies
from IPython.display import Audio
from IPython.core.display import display
#from ax.service.utils import report_utils
import shutil
import random
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import REMBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy
import torch
from ax.service.managed_loop import optimize


tri = 100
r_init = 5
f_rng = 5
randoo = [42]#, 103, 21, 600, 234, 1234, 66, 1000, 1589, 4142]
rand_flag = [0]
qq = [2, 7] #[5, 
sigg = [0.2, 10] #[1, 
dd = [4, 10, 20] #[4, 
DD = [2000] #

cnt = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#path1 = "/workspace/host/Experiments/Synthetic/Synthetic_perception/ALEBO/"
path1 = "GPR_files/bone/samples/Experiments/Synthetic/Synthetic_perception/ALEBO/hyper1/"

bp = []
val = []
exp = []
mo = []



def sc_evaluation_function(parameterization):
    global cnt
    x = np.array(list(parameterization.items()))[:,1].astype(float) 
    score_func = 0
    for i in range(D):
        score_func += math.floor(abs(x[i] + 0.5))**2
    return {"objective": (score_func, 0.0)}

def score_call():
    np.random.seed(1234)

    dat, fs = load("GPR_files/test.wav")
    sig_fft = stft(dat,n_fft=2*(D-1))
    w = np.loadtxt(os.path.join(path,"h.txt"))
    melfb = mel_frequencies(n_mels=D,fmax=fs/2)
    hh = np.interp(np.arange(0,D)/(D-1)*(fs/2), melfb, w)
    r = 10**(hh/10)
    fil = istft((sig_fft.T * r).T)
    wavfile.write(os.path.join(path,f"out{cnt+1}.wav"), fs, fil.astype(dat.dtype))
    display(Audio(fil.astype(dat.dtype),rate=fs))
    
    


for q in qq:        
    for sig in sigg:
        for d in dd:
            for D in DD:
                for rando in randoo:
                    
                    random.seed(rando)
                    torch.manual_seed(rando)
                    
                   
                    for no_rand in rand_flag:
                        if(no_rand==0):
                            no_f = [5]
                            mu_rng = np.vstack((np.arange(0,no_f[-1]),np.zeros(f_rng))).T
                        else:
                            no_f = [5, 15, 30]
                            mu_rng = np.vstack((np.array(random.sample(range(0, D), f_rng)),np.zeros(f_rng))).T
                        
                        if((D==2000 and d==4 and q==2 and sig==0.2) or(D==2000 and d==10 and q==2 and sig==0.2) or (D==2000 and d==20 and q==2 and sig==0.2)or (D==2000 and d==20 and q==2 and sig==10) or (D==2000 and d==10 and q==2 and sig==10) or (D==2000 and d==4 and q==2 and sig==10)or (D==2000 and d==4 and q==7 and sig==0.2)or (D==2000 and d==10 and q==7 and sig==0.2)or (D==2000 and d==20 and q==7 and sig==0.2)or (D==2000 and d==4 and q==7 and sig==10)or (D==2000 and d==10 and q==7 and sig==10)):
                            kkk = 0
                        else:    
                            
                            path = os.path.join(path1,f"exp_D{D}_d{d}_rng{5}_q{q}_sig{sig}_tri{tri}_no_f_{'_'.join(str(f) for f in no_f)}_rand{no_rand}_seed{rando}")
                            os.mkdir(path)
                            parameters = [
                                {"name": "x0", "type": "range", "bounds": [-5, 5.0], "value_type": "float"},
                                {"name": "x1", "type": "range", "bounds": [-5, 5.0], "value_type": "float"},
                            ]
                            parameters.extend([
                                {"name": f"x{i}", "type": "range", "bounds": [-5, 5.0], "value_type": "float"}
                                for i in range(2, D)
                            ])
    
    
                            for ex in no_f:
                                if(ex==0):
                                    q1=1
                                else:
                                    q1=q
                                mu = {}
                                for i in range(0, ex):
                                    mu[int(mu_rng[i,0])] = mu_rng[i,1]
                                print(mu,q1)
                                torch.manual_seed(rando)
    
                                abo_strategy = ALEBOStrategy(D=D, d=d, init_size=r_init,gp_kwargs={"q":q1,"mu":mu,"sig":sig,"device":device})
                                print(f"experiment start={ex}")
                                best_parameters, values, experiment, model = optimize(
                                    parameters=parameters,
                                    experiment_name=f"score_{D}",
                                    objective_name="objective",
                                    evaluation_function=sc_evaluation_function,
                                    minimize=True,
                                    total_trials=tri,
                                    random_seed=np.random.seed(rando),
                                    generation_strategy=abo_strategy,
                                );
                                bp.append(best_parameters)
                                val.append(values)
                                exp.append(experiment)
                                mo.append(model)
    
                                objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
                                np.savetxt(os.path.join(path,f"scores_{ex}.txt"),objectives)
                                np.savetxt(os.path.join(path,f"h_star_{ex}.txt"),np.array(list(best_parameters.items()))[:,1].astype(float))
                                samp = np.array([np.array(list(np.array([trial.arm.parameters for trial in experiment.trials.values()])[i].items()))[:,1].astype(float) for i in range(tri)])
                                np.savetxt(os.path.join(path,f"samp_{ex}.txt"),samp)
                                acq_val = np.array([experiment.trials[i].generator_run.gen_metadata['expected_acquisition_value'][0] for i in range(r_init,tri)])
                                np.savetxt(os.path.join(path,f"acq_val_{ex}.txt"),acq_val)
                                np.savetxt(os.path.join(path,f"mu_rng_{ex}.txt"),mu_rng)
                                f = open(os.path.join(path,f"kwargs_{ex}.txt"),"w")
                                f.write( str(np.array(list({"q":q1,"mu":mu,"sig":sig}.items()))) )
                                f.close()
    
