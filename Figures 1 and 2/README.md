Run `experiments.py` to run the main experiments for Figures 1 and 2, the results of which are saved in `results_A.csv`, `results_B.csv`, and `results_C.csv`. Run `population_benchmarks.ipynb` notebook to generate the population benchmarks for Figure 1, the results of which are saved in `benchmarks_A.csv` and `benchmarks_B.csv`. Finally, run `plots.ipynb` to reproduces the two plots for Figure 1 and the plot for Figure 2, saved as `fig1_plots.pdf` and `fig2_plot.pdf`.

We ran `experiments.py` with following Python and package versions:

- Python 3.11.16
- numpy==2.4.6
- pandas==3.0.6
- torch==2.14.1+cpu

The remaining scripts were run with the following Python and package versions:

- Python 3.14.9
- numpy==2.5.4
- pandas==3.1.0
- matplotlib==3.11.2
- cvxpy==1.9.3
- clarabel==0.11.1
- scs==3.3.1
- scipy==1.18.1
- jupyterlab==4.6.5