all:
	g++ -o main -fopenmp -O3 -Wall main.cpp
clean:
	rm main