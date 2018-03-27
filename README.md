# ShElastic: Spherical Harmonic Solution of Linear Elasticity Problem #

Fracture of ductile materials is mainly dominated by the nucleation, growth and coalescence of the voids in the material (Tvergaard, 1989). Recent studies on nano- and micron-sized voids revealed that the void growth is mainly dominated by the dislocations emitted from the voids or in the continuum body (Lubarda et al. 2004; Segurado et al. 2009, 2010; Traiviratana et al., 2008). To model the void-dislocation interaction system, dislocation dynamics (DD) simulation is a perfect tool. 

In DD simulation, the image stress solution of arbitrary-shaped boundary condition is usually solved by finite elements method (FEM). However, for some specific shaped boundaries such as cylindrical shape, we can utilize the symmetry of the shape and use analytical solution to improve the efficiency (Weinberger and Cai, 2007). In this project, we develop a semi-analytical solution based on spherical harmonic transformation ([SHTOOLS](https://shtools.oca.eu/shtools/)) to the traction boundary value problem that is applicable to any arbitrary traction loading on the spherical boundary for linear elastic material.

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary ()


* Version 0.2
* More about spherical harmonics ([wiki-spherical\_harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics))

### How do I get set up? ###

* This python package is based on python3, numpy, scipy, matplotlib, and shtools
* This configuration guideline is only tested on local Ubuntu 16.04LTS

1. Install python3 for ubuntu (more detailed instructions, please see python.org)
2. Install pip for python3 on ubuntu
    `sudo apt-get install python3-pip`
3. Install numpy, scipy, and matplotlib in python3 with pip (more detailed instructions on numpy.org)
4. Install SHTOOLS for python ([Instructions](https://shtools.oca.eu/shtools/www/install.html))

Then you are good to use the functions provided in this package.

* Running test cases interactively

1. Install Jupyter (see jupyter.org)
2. In terminal, 
    `cd ${WORK_DIR}/shelastic'
    'jupyter notebook'

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Wei Cai (caiwei@stanford.edu)
* Yifan Wang (yfwang09@stanford.edu)
* Xiaohan Zhang (xzhang11@stanford.edu)
