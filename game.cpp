#include <cstring>
#include <omp.h>
#include <cstdio>
#include <random>

#define NUM_HIDDEN_LAYER 3

const bool debug = false;
const unsigned short location_array[4][4] = {{12, 13, 14, 15}, {0, 4, 8, 12}, {0, 1, 2, 3}, {3, 7, 11, 15}};
std::mt19937 generator;
std::uniform_int_distribution <int> next_tile_gen(0, 10), location_gen(0, 3);

class GAME
{
	//Neural Network
	double hidden_layer_output[NUM_HIDDEN_LAYER + 1][16], output_layer[4];

	//Board
	unsigned short board[16], slide_dir[4];
public:
	GAME(): slide_dir({0, 1, 2, 3}) {}

	//Neural Network
	void get_NN_value(const double * weight)
	{
		double * last_layer, * current_layer;

		//initialize
		std::memset(hidden_layer_output, 0, sizeof(hidden_layer_output));
		std::memset(output_layer, 0, sizeof(output_layer));
		for(int i = 0 ; i < 16 ; i++) hidden_layer_output[0][i] = board[i];

		for(int i = 1 ; i <= NUM_HIDDEN_LAYER ; i++)
		{
			last_layer = hidden_layer_output[i-1];
			current_layer = hidden_layer_output[i];
			#pragma omp parallel for
			for(int j = 0 ; j < 16 ; j++) // current layer
				for(int k = 0 ; k < 16 ; k++) // last layer
					current_layer[j] += last_layer[k] * weight[((i-1) << 8) + (j << 4) + k];
		}
		#pragma omp parallel for
		for(int i = 0 ; i < 4 ; i++) //output layer
		{
			last_layer = hidden_layer_output[NUM_HIDDEN_LAYER];
			for(int j = 0 ; j < 16 ; j++) // nodes in the last hidden layer
				output_layer[i] += last_layer[j] * weight[(NUM_HIDDEN_LAYER << 8) + (i << 4) + j];
		}
		for(int i = 0 ; i < 4 ; i++) // sort to find the best move
			for(int j = i + 1 ; j < 4 ; j++) 
				if(output_layer[slide_dir[i]] < output_layer[slide_dir[j]]) std::swap(slide_dir[i], slide_dir[j]);
	}
	
	//Board
	int test(const double * weight)
	{
		unsigned next_tile, dir;
		std::memset(board, 0, sizeof(board));

		//initialize
		//"(location_gen(generator) << 2) + location_gen(generator)" is equal to rand[0, 16)
		next_tile = next_tile_gen(generator) ? 1 : 2;
		put_tile((location_gen(generator) << 2) + location_gen(generator), next_tile);
		next_tile = next_tile_gen(generator) ? 1 : 2;
		while(!put_tile((location_gen(generator) << 2) + location_gen(generator), next_tile));
		print_board();
		//start playing
		while(1)
		{
			dir = 4;
			get_NN_value(weight);
			for(int i = 0 ; i < 4 ; i++) if(slide(slide_dir[i])) {dir = slide_dir[i]; break;}
			if(debug) std::printf("move : %d\n", dir);
			print_board();
			if(dir & 4) break;
			next_tile = next_tile_gen(generator) ? 1 : 2;
			while(!put_tile(location_array[dir][location_gen(generator)], next_tile));
			print_board();
		}

		//count the score
		unsigned long long score = 0;
		for(int i = 0 ; i < 16 ; i++) score += std::pow(3, board[i] - 1);
		return score;
	}
	bool put_tile(const int location, const int tile) // if we can place the tile, do it and return true ; otherwise return false
	{
		if(board[location]) return false;
		board[location] = tile;
		return true;
	}
	bool slide_up()
	{
		bool change = false;
		#pragma omp parallel for
		for(int i = 0 ; i < 4 ; i++)
		{	
			int next_tile;
			for(int j = i ; j < 16 ; j += 4)
			{
				next_tile = j + 4;
				while(next_tile < 16 && !board[next_tile]) next_tile += 4;
				if(next_tile > 15) break;
				if(!board[j])
				{
					std::swap(board[j], board[next_tile]);
					change = 1;
					j -= 4;
					continue;
				}
				else if(board[j] == board[next_tile]) 
				{
					board[j]++;
					board[next_tile] = 0;
					change = 1;
				}
			}
		}
		return change;
	}
	bool slide_right()
	{
		bool change = false;
		#pragma omp parallel for
		for(int i = 3 ; i < 16 ; i += 4)
		{	
			int next_tile;
			for(int j = i ; j > i - 4 ; j--)
			{
				next_tile = j - 1;
				while(next_tile > i - 4 && !board[next_tile]) next_tile--;
				if(next_tile < i - 3) break;
				if(!board[j])
				{
					std::swap(board[j], board[next_tile]);
					change = 1;
					j++;
					continue;
				}
				else if(board[j] == board[next_tile]) 
				{
					board[j]++;
					board[next_tile] = 0;
					change = 1;
				}
			}
		}
		return change;
	}
	bool slide_down()
	{
		bool change = false;
		#pragma omp parallel for
		for(int i = 12 ; i < 16 ; i++)
		{	
			int next_tile;
			for(int j = i ; j > -1 ; j -= 4)
			{
				next_tile = j - 4;
				while(next_tile > -1 && !board[next_tile]) next_tile -= 4;
				if(next_tile < 0) break;
				if(!board[j])
				{
					std::swap(board[j], board[next_tile]);
					change = 1;
					j += 4;
					continue;
				}
				else if(board[j] == board[next_tile]) 
				{
					board[j]++;
					board[next_tile] = 0;
					change = 1;
				}
			}
		}
		return change;
	}
	bool slide_left()
	{
		bool change = false;
		#pragma omp parallel for
		for(int i = 0 ; i < 16 ; i += 4)
		{	
			int next_tile;
			for(int j = i ; j < i + 4 ; j++)
			{
				next_tile = j + 1;
				while(next_tile < i + 4 && !board[next_tile]) next_tile++;
				if(next_tile > i + 3) break;
				if(!board[j])
				{
					std::swap(board[j], board[next_tile]);
					change = 1;
					j--;
					continue;
				}
				else if(board[j] == board[next_tile]) 
				{
					board[j]++;
					board[next_tile] = 0;
					change = 1;
				}
			}
		}
		return change;
	}
	bool slide(const int dir) //try to slide first, return true if succeed
	{
		switch(dir)
		{
			case 0:
				return slide_up();
			case 1:
				return slide_right();
			case 2:
				return slide_down();
			case 3:
				return slide_left();
			default:
				printf("Error slide dirction.\n");
				exit(-1);
		}
		return false;
	}
	void print_board() // for debugging
	{
		if(!debug) return ;
		printf("-----------------\n");
		for(int i = 0 ; i < 4 ; i++)
				printf("| %d | %d | %d | %d |\n-----------------\n", board[(i << 2)], board[(i << 2) + 1], board[(i << 2) + 2], board[(i << 2) + 3]);
		printf("\n\n");
	}
};