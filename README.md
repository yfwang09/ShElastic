<!--
![LOGO](misc/logo.png)

[![Documentation](https://img.shields.io/badge/documentation-shtools.github.io%2FSHTOOLS%2F-yellow.svg)](https://shtools.github.io/SHTOOLS/)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.592762.svg)](https://doi-org.stanford.idm.oclc.org/10.1016/j.jmps.2019.01.020)
[![Paper](https://img.shields.io/badge/paper-10.1029/2018GC007529-orange.svg)](https://doi.org/10.1029/2018GC007529)
[![Join the chat at https://gitter.im/SHTOOLS/SHTOOLS](https://badges.gitter.im/SHTOOLS/SHTOOLS.svg)](https://gitter.im/SHTOOLS/SHTOOLS?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Twitter](https://img.shields.io/twitter/follow/pyshtools.svg?style=social&label=Follow)](https://twitter.com/intent/follow?screen_name=pyshtools)
-->

**This is a copy of our GitLab repository (https://gitlab.com/micronano_public/ShElastic). We create this copy to add Colab support for our examples.**

ShElastic is a python package of the numerical method that solves linear elasticity problem with displacement or traction boundary conditions on _spherical_ _interface_. The method is can be applied to both spherical void and solid sphere problem. The method is applied originally to solve spherical void problem of void-dislocation interaction in an infinite medium [1], and recently applied to solid sphere problem in cell-hydrogel interaction for measuring forces of the biological cell [2].

### FEATURES ###

* Fast transformation of spherical boundary condition between real and spherical harmonic space supported by [SHTOOLS](https://shtools.github.io/SHTOOLS/)
* Data structure of vectors (3x1) and tensors (3x3) in spherical harmonics representation
* Fast conversion between displacement, stress and traction field in spherical harmonic space
* Tutorials in python notebook and python scripts on how to use the ShElastic package to solve full elasticity problem with spherical boundary condition.

### EXAMPLES ###

The examples are implemented as jupyter notebooks in `notebook` folder. Example 1 - 4 are discussed in reference [1], example 5 and 6 are discussed in reference [2], and example 7 is for reference [3].

0. Spherical void in an infinite medium under hydrostatic pressure
1. Spherical void in an infinite medium under uniform tensile stress
2. Spherical void in an infinite medium near an infinite long straight screw dislocation
3. Spherical void in an infinite medium near a prismatic dislocation loop
4. Solid sphere under torsional force on the surface
5. Deformable hydrogel sphere under biological cellular forces
6. Deformable hydrogel sphere under opposite forces (test case)
7. "Teeth" indentation on the equator of a sphere

### Run Examples on Google Colab (Recommended) ###

* You only need to have a google account to be able to run the notebooks on Colab. All the package installation will be handled within the notebook.

1. Open a notebook in `examples` folder (Case00 - Case05)
2. Click the button ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) at the top to open the notebook in Colab.
3. Run the notebook by clicking `runtime/run all`(`Ctrl+F9`), or clicking the play button on the left of each cell to run the notebook cell-by-cell.

### Installation guide ###

* This python package is based on python3, numpy, scipy, matplotlib, jupyter and shtools
* This configuration guideline is only tested on Linux systems (Ubuntu 18.04LTS, CentOS 7)

1. Install python3.6 (more detailed instructions, please see python.org)
2. Install all required packages in python3 with pip:
    `python3 -m pip install --user numpy, scipy, matplotlib, ipython, jupyter, shtools`
3. View the examples by opening jupyter notebook:
    `cd $WORK_DIR/shelastic`
    `jupyter notebook`

Then in the webpage popped out, you can view and run all the examples in the `notebook` folder.

### Contribution guidelines ###

Please contact the authors if you have questions or want to contribute.

### Authors ###

* Yifan Wang (yfwang09@stanford.edu)
* Wei Cai (caiwei@stanford.edu)

### Related Articles ###

[1] Wang, Yifan, Xiaohan Zhang, and Wei Cai. "Spherical harmonics method for computing the image stress due to a spherical void." [Journal of the Mechanics and Physics of Solids 126 (2019): 151-167.](https://doi.org/10.1016/j.jmps.2019.01.020)

[2] Vorselen, Daan, Yifan Wang, Miguel M. de Jesus, Pavak K. Shah, Matthew J. Footer, Morgan Huse, Wei Cai, and Julie A. Theriot. "Microparticle traction force microscopy reveals subcellular force exertion patterns in immune cell–target interactions." [Nature communications 11, no. 1 (2020): 1-14.](https://doi.org/10.1038/s41467-019-13804-z)

[3] Vorselen, Daan, Barger, Sarah R., Wang, Yifan, Cai, Wei, Theriot, Julie A., Gauthier, Nils C., Krendel, Mira, "Phagocytic 'teeth' and myosin-II 'jaw' power target constriction during phagocytosis." [bioRxiv (2021)](https://www.biorxiv.org/content/10.1101/2021.03.14.435346v1.abstract).
