This repository contains the light curve and the code used to model
the VVV-WIT-08 object.

# Contents

* lc_model_opt.cpp, lc_model_opt.hpp, Makefile are the c++ codes used to compute fast occultation models of a star by an ellipse.
* lc_model_opt.py is a python wrapper around lc_model_opt.cpp

* lc_model_new.py is the code used to run sampling of the eclipse parameters using a variaty of samplers such as pymultinest/dynesty/zeus and others

# Usage

After running
$ make
to  compile the lc_model_opt.so library you should be able to
use the lc_model_opt.getlc to compute arbitrary light curves. The description of the arguments of the getlc() method are given in the .cpp file

# Authors

Sergey Koposov and Leigh Smith
