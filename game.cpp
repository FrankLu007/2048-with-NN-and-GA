#include <cstring>
#include <omp.h>
#include <cstdio>
#include <random>

//#define FNN // if FNN is not defined, it will use CNN automatically

#define NUM_HIDDEN_LAYER 3
#define POOL_SIZE 4

const bool debug = false;
const unsigned short location_array[4][4] = {{12, 13, 14, 15}, {0, 4, 8, 12}, {0, 1, 2, 3}, {3, 7, 11, 15}};
const unsigned short match[] = {3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12};
std::mt19937 generator;
std::uniform_int_distribution <int> next_tile_gen(0, 9), location_gen(0, 3);

class GAME
{
	//Neural Network
	#ifdef FNN
	double hidden_layer_output[NUM_HIDDEN_LAYER + 1][16];
	#else
	double hidden_layer_output[3][16];
	#endif

	double output_layer[4];

	//Board
	unsigned short board[16], slide_dir[4];
public:
	GAME(): slide_dir({0, 1, 2, 3}) {}

	//Neural Network
	#ifdef FNN
	void get_NN_value(const double * weight)
	{
		double * last_layer, * current_layer;
		unsigned short tmp_board[16], tmp_rotate[16];

		//initialize
		std::memset(output_layer, 0, sizeof(output_layer));
		for(int i = 0 ; i < 16 ; i++) tmp_board[i] = board[i];

		for(int rotate = 0 ; rotate < 8 ; rotate++)
		{
			std::memset(hidden_layer_output, 0, sizeof(hidden_layer_output));
			for(int i = 0 ; i < 16 ; i++) hidden_layer_output[0][i] = tmp_board[i];

			for(int i = 1 ; i <= NUM_HIDDEN_LAYER ; i++)
			{
				last_layer = hidden_layer_output[i-1];
				current_layer = hidden_layer_output[i];
				unsigned tmp = (i-1) * 17 * 16;
				#pragma omp parallel for
				for(int j = 0 ; j < 16 ; j++) // current layer
				{
					for(int k = 0 ; k < 16 ; k++) // last layer
						current_layer[j] += last_layer[k] * weight[tmp + (j * 17) + k];
					current_layer[j] += weight[tmp + (j + 1) * 17]; //bias
				}
			}
			#pragma omp parallel for
			for(int i = 0 ; i < 4 ; i++) //output layer
			{
				last_layer = hidden_layer_output[NUM_HIDDEN_LAYER];
				unsigned tmp = NUM_HIDDEN_LAYER * 17 * 16, dir = (i + rotate) & 3;
				for(int j = 0 ; j < 16 ; j++) // nodes in the last hidden layer
					output_layer[dir] += last_layer[j] * weight[tmp + (i * 17) + j];
				output_layer[dir] += weight[tmp + (i + 1) * 17]; // bias
			}

			//reflect
			if(i == 3) 
			{
				for(int i = 0 ; i < 8 ; i++) 
					if(i < 4) std::swap(tmp_board[i], tmp_board[i + 12]);
					else std::swap(tmp_board[i], tmp_board[i + 4]);
				std::swap(output_layer[0], output_layer[2]);
			}

			// rotate clockwise
			for(int i = 0 ; i < 16 ; i++) tmp_rotate[i] = tmp_board[i];
			for(int i = 0 ; i < 16 ; i++) tmp_board[match[i]] = tmp_rotate[i];
		}

		std::swap(output_layer[0], output_layer[2]);

		for(int i = 0 ; i < 4 ; i++) // sort to find the best move
			for(int j = i + 1 ; j < 4 ; j++) 
				if(output_layer[slide_dir[i]] < output_layer[slide_dir[j]]) std::swap(slide_dir[i], slide_dir[j]);
	}
	#else
	double matrix_multiple(const unsigned short loc, const unsigned short layer, const double * pool) //loc : location of start point
	{
		double sum = 0;
		const unsigned short length = 4 - layer;
		for(int i = 0 ; i < 2 ; i++) for(int j = 0 ; j < 2 ; j++) for(int k = 0 ; k < 2 ; k++) 
			sum += hidden_layer_output[layer][i * length + k] * pool[k * 2 + j];
		return sum;
	}
	void get_NN_value(const double * weight)
	{
		std::memset(hidden_layer_output, 0, sizeof(hidden_layer_output));
		std::memset(output_layer, 0, sizeof(output_layer));
		for(int i = 0 ; i < 16 ; i++) hidden_layer_output[0][i] = board[i];

		for(int i = 0 ; i < 3 ; i++) for(int j = 0 ; j < 3 ; j++)
			hidden_layer_output[1][i * 3 + j] += matrix_multiple(i * 4 + j, 0, weight);
		for(int i = 0 ; i < 2 ; i++) for(int j = 0 ; j < 2 ; j++)
			hidden_layer_output[2][i * 2 + j] += matrix_multiple(i * 3 + j, 1, weight + 4);
		for(int i = 0 ; i < 2 ; i++) for(int j = 0 ; j < 2 ; j++)
			output_layer[i] += matrix_multiple(i * 2 + j, 2, weight + 8);

		for(int i = 0 ; i < 4 ; i++) // sort to find the best move
			for(int j = i + 1 ; j < 4 ; j++) 
				if(output_layer[slide_dir[i]] < output_layer[slide_dir[j]]) std::swap(slide_dir[i], slide_dir[j]);
	}
	#endif
	
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