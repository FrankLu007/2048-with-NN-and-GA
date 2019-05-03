#include <cstring>
#include <random>
#include "game.cpp"

#define NUM_TEST 100

const unsigned gene_length = 256 * NUM_HIDDEN_LAYER + 64; // 3 hidden layers + 1 output layer

GAME game;
std::mt19937_64 generator(time(NULL));

class GENE
{
public:
	double gene[gene_length];
	double score;
	GENE(const bool test = true)
	{
		std::memset(gene, 0, sizeof(gene));
		if(test) score = game.test(gene, NUM_TEST);
	}
	void crossover(const double * p1, const double * p2)
	{

	}
	void mutation(const double * p, const double var)
	{
		std::normal_distribution <double> mutation_gen(0, var);
		for(unsigned i = 0 ; i < gene_length ; i++) gene[i] = P[i] + var * mutation_gen(generator);
		score = game.test(gene, NUM_TEST);
	}
}