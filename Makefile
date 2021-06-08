#EIGEN_PATH=-I/usr/include/eigen3/ 
OPT=-march=native -mtune=native -O3 -g 
#OPT=-O0 -g
FLAGS= $(OPT) -std=c++14 -fPIC
OPENMP=-fopenmp
LINK=-lgomp -lm
#OPENMP=
#FLAGS=-O0 -g -ggdb

all: liblc_model_opt.so
lc_model_opt.o: lc_model_opt.cpp
	g++ $(OPENMP) -Wall lc_model_opt.cpp -lm $(FLAGS) -c -fPIC
liblc_model_opt.so: lc_model_opt.o
	g++ $(FLAGS) -shared lc_model_opt.o $(LINK) -o liblc_model_opt.so
lc_model_opt_hole.o: lc_model_opt_hole.cpp
	g++ $(OPENMP) -Wall lc_model_opt_hole.cpp -lm $(FLAGS) -c -fPIC

test: lc_model_opt.o test.cpp
	g++ $(OPENMP) -Wall lc_model_opt.o test.cpp -lm $(FLAGS) -o test
timer: lc_model_opt.o timer.cpp
	g++ $(OPENMP) -Wall lc_model_opt.o timer.cpp -lm $(FLAGS) -o timer
clean:
	rm -f *.o *.so a.out
