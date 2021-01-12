CXXFLAGS += -std=c++14 -mtune=native -march=native -Wall -O3 -DNDEBUG -g -fopenmp
LIBS += -lrt

all: main test

main: main.cpp
	g++ $(CXXFLAGS) main.cpp -o main

test: test.cpp
	g++ $(CXXFLAGS) test.cpp -o test

clean:
	rm -rf main *.o
