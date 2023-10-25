# PSST (Polymer Solution Scaling Theory) Library

The `psst` library allows users to study and implement our deep learning scaling theory methods and test similar approaches.

## `psst` Module

The novel parts of the code are grouped into the module `psst`, which will be added to PyPI and conda in the near future. For now, the library can be installed using the command `pip install .` in the head directory (the directory containing `pyproject.toml`). Dependencies will be handled by pip. Users may wish to create a new virtual environment first.

- The `models` submodule contains two convolutional neural network (CNN) models we used in our initial research, `models.Inception3` and `models.Vgg13`.
- `psst.configuration` defines the reading and structure of the system configuration, as specified in a YAML or JSON file.
- `psst.samplegenerator` contains the `SampleGenerator` class that is used to procedurally generate batches of viscosity curves as functions of concentration (`phi`) and chain degree of polymerization (`Nw`). The submodule also contains `normalize` and `unnormalize` functions. Normalizing transforms the true values, optionally on a log scale, to values between 0 and 1.
- `psst.training` contains the `train` and `validate` functions, as well as checkpointing functionality for the model and optimizer.

NOTE: The normalize/unnormalize functions and the checkpointing functionality may move to different submodules, perhaps new files.

## Other Directories

The `examples` directory contains scripts to optimize and train networks, and one to evaluate experimental data. These are similar to the scripts used during our research. Details are in `examples/README.md`.

The `docs` directory contains documentation written using Sphinx.

The `img` directory will contain images of plots/figures used in this README and in the `derivations.md` file.

## Future Plans

The plan for this library is for a user to train a neural network using procedurally generated data from `SampleGenerator`. They are free to train an independently developed model, modify any parameters such as the "resolution" of the data (the generated viscosity data is represented as a 2D image), and use the resulting models to evaluate real, experimental specific viscosity data. 

However, we recognize that not every researcher has the time or patience to learn about machine learning algorithms and our complicated analysis methods. Therefore, we plan to publish a web app that will allow researchers to upload their specific viscosity data in a .csv format, enter pertinent information about the units of measure and the monomer used, and receive results in the form of plotted data and .csv formatted files that give reasonable estimates of Kuhn length in solution, thermal blob size, degree of polymerization between entanglements as a function of concentration, and several more key solution properties. An announcement will be made on this GitHub page as well as the Dobrynin Group [website](https://dobryninlab.unc.edu).
