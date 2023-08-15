# OracleBO Dataset
## Evaluation Parameters
In our experiments, we optimize the objective functions for:

- f_evals = 100 function evaluations

- r\_init = 5 initial random samples after which the acquisition sampling begins

- D = 2000 is the high-dimensional space

- d = 4 is embedding space for $P1, P2, P3$, Branin and Rosenbrock

- d = 6 for Hartmann6 

- Branin, Rosenbrock, and Hartmann6 have effective dimensionality d_e = 2, 4, 6, respectively. 

- Different numbers of Oracle queries L = {0, 1, 3, 5, 15, 30} are used and L=0 implies no Oracle queries are available and OracleBO functions as just ALEBO (Letham et al.2020)

- q=5 acquisition samples are used in Batch Acquisition function (BAF)


- In the Dimension Matched Sampler (DMS), we use variance sigma = 1

- For Branin, we use the minimizer (pi,2.275) to generate Q_d queries and for perception functions P1,P2,P3 we use the minimizer h_i^*=0, i=1,2,....,N. For Hartmann6 and Rosenbrock we use their unique minimizers.

- We run 10 random runs of each experiment
    
