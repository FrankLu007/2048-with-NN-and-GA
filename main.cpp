#include <cstdio>
#include <cstring>
#include <random>
#include <omp.h>
#include "game.cpp"


#define NUM_TEST_LIMIT 100
#define ITERATION 10000000

const unsigned gene_length = (NUM_HIDDEN_LAYER << 8) + 64; // 3 hidden layers + 1 output layer

extern std::mt19937 generator;
std::normal_distribution <double> mutation_gen(0, 1);
GAME game[NUM_TEST_LIMIT]; //disable to parallel

class GENE
{
	double gene[gene_length];
	double score;
public:
	GENE() : score(-1) {}
	double get_score() {return score;}
	double count_score(int time = NUM_TEST_LIMIT)
	{
		if(time > NUM_TEST_LIMIT) {printf("Error test number.\n"); exit(-1);}
		unsigned long long sum = 0;

		#pragma omp parallel for
		for(int i = 0 ; i < time ; i++) 
		{
			unsigned long long tmp = game[i].test(gene);
			#pragma omp critical
			sum += tmp;
		}

		return score = (double)sum / time;
	}
	void crossover(const double * p1, const double * p2) {}
	void mutation(const GENE & p, const double var)
	{
		#pragma omp parallel for
		for(unsigned i = 0 ; i < gene_length ; i++) gene[i] = p.gene[i] + var * mutation_gen(generator);
	}
	void print(int l)
	{
		for(int i = 0 ; i < l ; i++) printf("%lf ", gene[i]);
		printf("\n");
	}
};

int main(int N, char ** args)
{
	if(N != 3) 
	{
		std::printf("Error format.\n"); 
		exit(-1);
	}
	GENE parent, child;
	double var = 0.1, alpha = std::atof(args[2]);
	unsigned long long mutation_success = 0, section = std::atoll(args[1]);

	generator.seed(time(NULL));
	parent.count_score();
	printf("0 : %lf\n", parent.get_score());

	for(unsigned long long iteration = 1 ; iteration < ITERATION ; iteration++)
	{
		child.mutation(parent, var);
		if(child.count_score() > parent.get_score()) 
		{
			std::swap(parent, child);
			mutation_success++;
			std::printf("%llu : %lf\n", iteration, parent.get_score());
		}
		if(!(iteration % section))
		{
			if(mutation_success * 5 > section) var /= alpha;
			else if(mutation_success * 5 < section) var *= alpha;
			mutation_success = ;
		}
	}
	return 0;
}