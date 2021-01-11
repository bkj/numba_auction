CXXFLAGS += -std=c++11 -mtune=native -march=native -Wall -O3 -DNDEBUG -g -fopenmp
LIBS += -lrt

all: auction

apsp: auction.cpp
	g++ $(CXXFLAGS) -o auction auction.cpp

clean:
	rm -rf auction *.o
