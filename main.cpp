#include <cstdio>
#include <cstring>
#include <random>
#include <omp.h>
#include "game.cpp"

#define NUM_TEST_LIMIT 10000
#define ITERATION 500

#ifdef FNN
const unsigned gene_length = (NUM_HIDDEN_LAYER << 8) + 64; // 3 hidden layers + 1 output layer
#else
const unsigned gene_length = 12; // 3 pool matrix with each is 2 x 2
#endif

extern std::mt19937 generator;
std::normal_distribution <double> mutation_gen(0, 1);
GAME game[NUM_TEST_LIMIT]; //disable to parallel

class GENE
{
	double gene[gene_length];
	double score;
public:
	GENE() : score(-1) {}
	GENE(FILE * fp)
	{
		for(int i = 0 ; i < gene_length ; i++) fscanf(fp, "%lf", &gene[i]);
		score = -1;
	}
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
	void print(int l = gene_length)
	{
		std::FILE * fp = std::fopen("gene.txt", "w");
		for(int i = 0 ; i < l ; i++) std::fprintf(fp, "%lf ", gene[i]);
		std::fprintf(fp, "\n");
	}
};

int main(int N, char ** args)
{
	if(N < 3) 
	{
		std::printf("Error format.\n"); 
		exit(-1);
	}
	GENE parent, child;
	double var = 0.1, alpha = std::atof(args[2]);
	unsigned long long mutation_success = 0, section = std::atoll(args[1]);

	if(N == 4) parent = GENE(std::fopen(args[3], "r"));

	generator.seed(time(NULL));
	parent.count_score();
	printf("Initial value : %lf %lf\n", parent.get_score(), var);

	for(unsigned long long iteration = 1 ; iteration < ITERATION ; iteration++)
	{
		child.mutation(parent, var);
		if(child.count_score() > parent.get_score()) 
		{
			std::swap(parent, child);
			mutation_success++;
			std::printf("Iteration : %5llu : %lf %lf\n", iteration, parent.get_score(), var);
		}
		if(!(iteration % section))
		{
			if(mutation_success * 5 > section) var /= alpha;
			else if(mutation_success * 5 < section) var *= alpha;
			var = std::max(0.0001, var);
			mutation_success = 0;
		}
		if(iteration % 100 == 0) std::printf("Iteration : %5llu %lf\n", iteration, var);
	}
	parent.print();
	return 0;
}