# Setup

To explore different software solutions, we can use Jupyter notebooks that separate individual tasks. Using a Notebook allows us to document and give additional explanation all in the same place. All within a browser. To setup, you will need to install Anaconda and Jupyter Notebooks. Once installed, you can follow these steps to get a notebook running.

(Assuming you are on Windows)

1) Open up a Anaconda Prompt and navigate to this folder (Bandbuddy/ML)

2) Run `conda env create -f environment.yml` (Note: This step only has to be done once on a given machine)

3) Run `conda activate BB`. You should now see (BB) on the left side of your command-line. 

Note: These first 3 steps set up a Python environment with the packages specified in `environment.yml`. This allows the `import` statements in the notebooks to work. If a import is failing, make sure the package is listed in the environment file.

4) Run `jupyter notebook`. This will start a jupyter server and open up a web-page which exposes the current directory. You can then open up any notebook file `.IPYNB` 

