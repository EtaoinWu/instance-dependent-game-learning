# Instance-dependent bandit learning in games

This is the code repository for the preprint
*Instance-dependent regret bounds for learning
two-player zero-sum games with bandit feedback*, authored
by Shinji Ito, Haipeng Luo, Taira Tsuchiya, and Yue Wu.

## Structure

The two experiments in the article each correspond to a
`.ipynb` file under the root folder. They each generate
their result as folders or `.json` files, and generate
the corresponding figures both within the jupyter notebook
and as `.pdf` files.

The algorithms are implemented in `onlineax/` as a Python
module.

## Dependencies
The `env.yml` specifies the conda environment used in our
experiments. Use the following command to install it:

```bash
conda env create -f env.yml
# or:
mamba env create -f env.yml
# or:
micromamba env create -f env.yml
```

The code should run on a clean install of the environment.

## Reproducibility

The code runs on a x64 Linux machine with 64 CPU cores and 
128GB of memory. Due to the usage of `jax.random`, the 
pseudo-random number generator (PRNG) is seeded completely
deterministic.
