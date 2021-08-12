This is our experiments directory. There are two main scripts at the moment (am still working on memory mapping on DRAM script). These scripts are:

1) ./mnist_dram.py
2) ./mnist_pymm.py

These scripts will run a SWAG (SWA-Gaussian) experiment using either pymm or dram. Both scripts take a few optional arguments and only one required argument. The required argument is the model string (aka class name of the model in lower-case).

Models in these experiments are classified based on the size of the posterior they create. Most models use Gbs of memory, however one model will use Tbs. You can see which models are available if you use the "-h" or "--help" flags when running the script. I would also recommend doing this anyways to see what optional arguments are available.

Let's say you want to run an experiment using DRAM and a model which takes 1.5Gb of memory for the posterior (i.e. the DRAM experiment from the extended abstract). You could use the following command to train a single epoch (using default values for the other optional arguments):


```python mnist_dram.py -e 1 model_1_5gb```

