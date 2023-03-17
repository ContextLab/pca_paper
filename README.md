# Overview

This repository contains code and data pertaining to [High-level cognition is supported by information-rich but compressible brain activity patterns](link) by Lucy L. W. Owen and Jeremy R. Manning.

The repository is organized as follows:

```
root
├── code: analysis code used in the paper
│   └── notebooks: run these to reproduce all figures and statistical tests
├── data: all data analyzed in the paper
│   ├── images: example images used/analyzed in Figure 1
│   ├── networks: brain parcellation-related files
│   ├── neurosynth: topic decoding-related files
│   └── scratch: pre-computed results and temporary (*.pkl) files
└── paper: all files needed to generate a PDF of the paper and supplement
    ├── compile.sh: run this to generate PDFs (requires a latex distribution with pdflatex and bibtex support)
    └── figs: PDF copies of all figures
```

Our project uses [davos](https://github.com/ContextLab/davos) to improve shareability and compatability across systems.

# Setup instructions

Note: we have tested these instructions on MacOS and Ubuntu (Linux) systems.  We *think* they are likely to work on Windows systems too, but we haven't explicitly verified Windows compatability.

We recommend running all of the analyses in a fresh Python 3.10 conda environment.  To set up your environment:
  1. Install [Anaconda](https://www.anaconda.com/)
  2. Clone this repository by running the following in a terminal: `git clone https://github.com/ContextLab/pca_paper.git`  3. Create a new (empty) virtual environment by running the following (in the terminal): `conda create --name pca-paper python=3.10` (follow the prompts)
  3. Navigate (in terminal) to the activate the virtual environment (`conda activate pca-paper`)
  4. Install support for jupyter notebooks (`conda install -c anaconda ipykernel jupyter`) and then add the new kernel to your notebooks (`python -m ipykernel install --user --name=pca-paper`).  Follow any prompts that come up (accepting the default options should work).
  5. Navigate to the `code` directory (`cd code`) in terminal
  6. Start a notebook server (`jupyter notebook`) and click on the notebook you want to run in the browser window that comes up.  The `pie_images.ipynb` notebook is a good place to start.  Selecting "Restart & Run All" from the "Kernel" menu will automatically run all cells.
  7. When you're running the notebooks, always make sure the notebook kernel is set to `pca-paper` (indicated in the top right).  If not, in the `Kernel` menu at the top of the notebook, select "Change kernel" and then "pca-paper".
  8. To stop the server, send the "kill" command in terminal (e.g., `ctrl` + `c` on a Mac or Linux system).
  9. To "exit" the virtual environment, type `conda deactivate`.

Notes:
- After setting up your environment for the first time, you can skip steps 1, 2, 3, and 5 when you wish to re-enter the analysis environment in the future.
- After running the `decoding_and_compression.ipynb` notebook fully, the correct versions of all required packages for that notebook *and the other notebooks* will be automatically installed.  To run any other notebook:
  - Select the desired notebook from the Jupyter "Home Page" menu to open it in a new browser tab
  - Verify that the notebook is using the `pca-paper` kernel, using the above instructions to adjust the kernel if needed.
  - Select "Kernel" $\rightarrow$ "Restart & Run All" to execute all of the code in the notebook.

To remove the `pca-paper` environment from your system, run `conda remove --name pca-paper --all` in the terminal and follow the prompts.  (If you remove the `pca-paper` environment, you will need to repeat the initial setup steps if you want to re-run any of the code in the repository.)

Each notebook contains embedded documentation that describes what the various code blocks do.  Any figures you generate will end up in `paper/figures/source`.  Statistical results are printed directly from the notebooks when you run them.
