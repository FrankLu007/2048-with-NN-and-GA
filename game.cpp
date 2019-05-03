#include <cstring>
#include <cstdio>
#include <random>
#include "neural_network.cpp"

const unsigned short location_array[][] = {{12, 13, 14, 15}, {0, 4, 8, 12}, {0, 1, 2, 3}, {3, 7, 11, 15}};

class GAME
{
public:
	std::mt19937 generator(time(NULL));
	std::uniform_distribution <int> next_tile_gen(0, 10), location_gen(0, 16);
	int board[16];
	double test(const double * gene, const int time) // test for several times
	{
		int sum = 0;
		#pragma omp parallel for
		for(int i = 0 ; i < time ; i++) 
		{
			int tmp = test(gene);
			#pragma omp critical
			sum += tmp;
		}
		return (double)sum / time;
	}
	int test(const double * gene) // test only once
	{
		Neural_Network NN(gene, board);
		int next_tile, move_list[4], dir, score = 0;
		std::memset(board, 0, 64);

		//initialize
		next_tile = next_tile_gen(generator) ? 1 : 2;
		board[location_gen(generator)] = next_tile;
		next_tile = next_tile_gen(generator) ? 1 : 2;
		while(!put_tile(location_gen(generator), next_tile));

		//start playing
		while(1)
		{
			dir = -1;
			NN.get_NN_value(gene, board, move_list);
			for(int i = 0 ; i < 4 ; i++) if(move(move_list[i])) {dir = i; break;}
			shuffle(location_array[dir], location_array[dir] + 4, generator);
			next_tile = next_tile_gen(generator) ? 1 : 2;
			for(int i = 0 ; i < 4 ; i++) if(put_tile(location_array[dir][i], next_tile)) break;
		}

		//count the score
		for(int i = 0 ; i < 16 ; i++) score += std::pow(3, board[i] - 1);
		return score;
	}
	bool put_tile(const int location, const int tile) // if we can place the tile, do it and return true ; otherwise return false
	{
		if(board[location]) return false;
		board[location] = tile;
		return true;
	}
}