CXXFLAGS += -std=c++14 -mtune=native -march=native -Wall -O3 -DNDEBUG -g -fopenmp
LIBS += -lrt

all: main

main: main.cpp auction.hpp
	g++ $(CXXFLAGS) main.cpp -o main

clean:
	rm -rf main *.o
