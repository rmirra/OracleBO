# Evaluation Parameters
In our experiments, we have:

- We have three volunteers $U1,U2,U3$

- $f\_{evals} = 25$ function evaluations

- $r\_{init} = 5$ initial random samples after which the acquisition sampling begins

- $D = 512, \mathbf{R}^N$ is the high-dimensional space

- $d = 4, \mathbf{R}^d$ is embedding space


- Different numbers of $Q_d$ queries $L = {1, 3, 5}$ are used and $L=0$ implies no $Q_d$ queries are available and OracleBO functions as just ALEBO (Letham et al.2020)

- $q=5$ acquisition samples are used in Batch Acquisition function (BAF)

- In the Dimension Matched Sampler (DMS), we use variance $\sigma = 1$



    
