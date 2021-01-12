CXXFLAGS += -std=c++14 -mtune=native -march=native -Wall -O3 -DNDEBUG -g -fopenmp
LIBS += -lrt

all: main test cutest

main: main.cpp
	g++ $(CXXFLAGS) main.cpp -o main

test: test.cpp
	g++ $(CXXFLAGS) test.cpp -o test

cutest: cutest.cu
	nvcc -O3 --compiler-options "$(CXXFLAGS)" cutest.cu -o cutest

clean:
	rm -rf main test cutest *.o
