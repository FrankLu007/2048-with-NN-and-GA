#define NUM_HIDDEN_LAYER 3

class Neural_Network
{
public:
	double hidden_layer_output[NUM_HIDDEN_LAYER + 1][16];
	double output_layer[4];
	double * gene;
	int * board;
	Neural_Network(const double * _gene, const int * _board)
	{
		gene = _gene;
		board = _board;
	}
	void get_NN_value(int * move_list);
	{
		double * last_layer, * current_layer;
		memset(hidden_layer_output, 0, sizeof(hidden_layer_output));
		for(int i = 0 ; i < 16 ; i++) hidden_layer_output[0][i] = board[i];
		for(int i = 1 ; i <= NUM_HIDDEN_LAYER ; i++)
		{
			last_layer = hidden_layer_output[i-1];
			current_layer = hidden_layer_output[i];
			#pragma omp parallel for
			for(int j = 0 ; j < 16 ; j++)
			{
				for(int k = 0 ; k < 16 ; k++) current_layer[j] += last_layer[k] * gene[((i-1) << 8) + (j << 4) + k]
			}
		}
	}
}