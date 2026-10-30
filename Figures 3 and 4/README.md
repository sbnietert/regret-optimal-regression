To produce Figures 3 and 4, use Python 3.12 or newer and run the commands below from this folder.

1. Install the dependencies:

   ```sh
   python -m pip install -r requirements.txt
   ```

2. Download [institutions.tar.gz from Zenodo](https://zenodo.org/records/13382427/files/institutions.tar.gz?download=1)
   (about 479 MB) and extract it into this folder:

   ```sh
   tar -xzf /path/to/institutions.tar.gz
   ```

   Replace `/path/to/institutions.tar.gz` with the downloaded file's path.
   The result must include `institutions/agg_1_hour/0.csv` alongside the other
   institution CSVs. Only `agg_1_hour` is used; the other aggregation folders
   are not needed. Data: [CESNET-TimeSeries24](https://doi.org/10.5281/zenodo.13382427),
   Koumar, Hynek, Čejka, and Šiška, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).


3. Run the experiment:

   ```sh
   python experiments.py
   ```

   This writes `results_cesnet.csv` and `target_geometry_cesnet.csv`.

4. Open the plotting notebook and run all its cells:

   ```sh
   jupyter lab plots.ipynb
   ```

   It saves `fig3_plot` and `fig4_plots` as PDF files in this folder.

