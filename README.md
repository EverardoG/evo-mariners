# Quickstart

Install [conda](https://docs.conda.io/en/latest/). Run the following commands.

```
conda create -n mariners
conda activate mariners
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install pagmo pagmo-devel

git clone git@github.com:EverardoG/evo-mariners.git
cd evo-mariners
.build.sh
```

# Slow-start

This repository depends on [pagmo](https://esa.github.io/pagmo2/install.html). I would recommend installing this dependency using [conda](https://docs.conda.io/en/latest/). After installing conda, you can create a conda environment for this project with the following commands.

```
conda create -n mariners
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install pagmo pagmo-devel
```

This will make a new conda environment called "evo-mariners" that has the necessary dependencies. To use this environment, all you have to do is activate it with the following command.
```
conda activate mariners
```

If you want to deactivate this environment, you can use the following command.
```
conda deactivate
```

In order to build the code, you can simply run `build.sh` from the project directory. If you are using conda to manage pagmo, then make sure you activate your conda environment before building. This is what the necessary commands look like if you are using conda, and you installed `evo-mariners` in your home directory.
```
conda activate mariners
cd ~/evo-mariners
./build.sh
```

The build files will be found in the `~/evo-mariners/build` folder.
