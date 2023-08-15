# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:48:05 2023

@author: rmirr
"""

import numpy as np
import math
from ax.utils.measurement.synthetic_functions import branin
import os
from scipy.io import wavfile
from librosa import stft, istft, load, mel_frequencies
from IPython.display import Audio
from IPython.display import display
#from ax.service.utils import report_utils
import shutil
import random
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import REMBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy
import torch
from ax.service.managed_loop import optimize

random.seed(42)

D = 2000
d = 20
tri = 100
r_init = 5
f_rng = 30
no_f = [1, 2, 3, 4, 5, 15, 30]
cnt = 0
q = 5
sig = 1
mu_rng = np.vstack((np.array(random.sample(range(0, D), f_rng)),np.zeros(f_rng))).T
mu_rng1 = np.vstack((np.arange(0,no_f[-1]),np.zeros(f_rng))).T

path1 = "/workspace/host"
path = os.path.join(path1,f"exp_D{D}_d{d}_rng{5}_q{q}_sig{sig}_tri{tri}_no_f_{'_'.join(str(f) for f in no_f)}_rand")
os.mkdir(path)

def sc_evaluation_function(parameterization):
    global cnt
    x = np.array(list(parameterization.items()))[:,1].astype(float)
    
    #x = np.array([parameterization["x0"], parameterization["x1"], parameterization["x2"], parameterization["x3"], parameterization["x4"], parameterization["x5"], parameterization["x6"], parameterization["x7"], parameterization["x8"], parameterization["x9"], parameterization["x10"], parameterization["x11"]])
    #, parameterization["x12"], parameterization["x13"], parameterization["x14"], parameterization["x15"], parameterization["x16"], parameterization["x17"], parameterization["x18"], parameterization["x19"]])
    #print(x)    
    #b1 = 5.1 / (4 * math.pi ** 2) 
    #c1 = 5 / math.pi
    #r1 = 6
    #t1 = 1 / (8 * math.pi)
    #score_func = (x[1] - b1*(x[0]**2) + c1*x[0] - r1)**2 + 10*(1-t1)*(math.cos(x[0])) + 10
    
    #np.savetxt("GPR_files/bone/samples/h.txt",x)
    #score_call()
    #score_func = float(input())
    #cnt = cnt + 1
    #os.rename("GPR_files/bone/samples/h.txt",f"GPR_files/bone/samples/h{cnt}.txt" )
    
    score_func = 0
    for i in range(D):
        score_func += math.floor(abs(x[i] + 0.5))**2
    
    #score_func = math.floor(abs(x[0] + 0.5))**2 + math.floor(abs(x[1] + 0.5))**2 + math.floor(abs(x[2] + 0.5))**2 + math.floor(abs(x[3] + 0.5))**2 + math.floor(abs(x[4] + 0.5))**2 + math.floor(abs(x[5] + 0.5))**2+ math.floor(abs(x[6] + 0.5))**2 + math.floor(abs(x[7] + 0.5))**2 + math.floor(abs(x[8] + 0.5))**2 + math.floor(abs(x[9] + 0.5))**2 + math.floor(abs(x[10] + 0.5))**2 + math.floor(abs(x[11] + 0.5))**2
    #+ math.floor(abs(x[12] + 0.5))**2 + math.floor(abs(x[13] + 0.5))**2 + math.floor(abs(x[i] + 0.5))**2 + math.floor(abs(x[14] + 0.5))**2 + math.floor(abs(x[15] + 0.5))**2 + math.floor(abs(x[16] + 0.5))**2 + math.floor(abs(x[17] + 0.5))**2 + math.floor(abs(x[18] + 0.5))**2 + math.floor(abs(x[19] + 0.5))**2
    #, "fsamp": (np.linalg.norm(x-0.5*np.ones(x.shape[0])), 0.0)
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

parameters = [
    {"name": "x0", "type": "range", "bounds": [-5, 5.0], "value_type": "float"},
    {"name": "x1", "type": "range", "bounds": [-5, 5.0], "value_type": "float"},
]
parameters.extend([
    {"name": f"x{i}", "type": "range", "bounds": [-5, 5.0], "value_type": "float"}
    for i in range(2, D)
])

torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bp = []
val = []
exp = []
mo = []

for ex in no_f:
    if(ex==0):
        q1=1
    else:
        q1=q
    mu = {}
    for i in range(0, ex):
        mu[int(mu_rng[i,0])] = mu_rng[i,1]
    print(mu,q1)
    torch.manual_seed(42)
    
    abo_strategy = ALEBOStrategy(D=D, d=d, init_size=r_init,gp_kwargs={"q":q1,"mu":mu,"sig":sig,"device":device})
    print(f"experiment start={ex}")
    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        experiment_name="score_512",
        objective_name="objective",
        evaluation_function=sc_evaluation_function,
        minimize=True,
        total_trials=tri,
        random_seed=np.random.seed(42),
        generation_strategy=abo_strategy,
        #torch_device = device
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
    np.savetxt(os.path.join(path,f"min_{ex}.txt"),np.array(list(best_parameters.items()))[:,1].astype(float))
    np.savetxt(os.path.join(path,f"mu_rng_{ex}.txt"),mu_rng)
    f = open(os.path.join(path,f"kwargs_{ex}.txt"),"w")
    f.write( str(np.array(list({"q":q,"mu":mu,"sig":sig}.items()))) )
    f.close()