#include <cstdio>
#include <cstring>
#include <random>
#include <omp.h>
#include "game.cpp"

#define NUM_TEST_LIMIT 10000
#define ITERATION 100

//#define FNN

#ifdef FNN
const unsigned gene_length = (NUM_HIDDEN_LAYER * 17 * 16) + 68; // 3 hidden layers + 1 output layer
#else
const unsigned gene_length = 32; // 3 pool matrix with each is 2 x 2 and 20 for FNN
#endif

extern std::mt19937 generator;
std::normal_distribution <double> mutation_gen(0, 1);
GAME game[NUM_TEST_LIMIT]; //disable to parallel

class GENE
{
	double gene[gene_length];
	double score;
public:
	GENE() : score(-1) 
	{
		for(int i = 0 ; i < gene_length ; i++) gene[i] = mutation_gen(generator);
	}
	GENE(FILE * fp)
	{
		for(int i = 0 ; i < gene_length ; i++) std::fscanf(fp, "%lf", &gene[i]);
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
	void crossover(const GENE & p1, const GENE & p2) 
	{
		unsigned loc = mutation_gen(generator) * gene_length;
		loc %= gene_length;
		for(int i = 0 ; i < gene_length ; i++) gene[i] = i < loc ? p1.gene[i] : p2.gene[i];
	}
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
	GENE parent1, parent2, child;
	double var = 0.1, alpha = std::atof(args[2]), tmp;
	unsigned long long mutation_success = 0, section = std::atoll(args[1]);

	if(N == 4) parent1 = GENE(std::fopen(args[3], "r"));

	generator.seed(time(NULL));
	parent1.count_score();
	parent2.count_score();
	std::fprintf(stderr, "Iteration : %5llu %lf %lf %lf\n", 0, var, std::log(parent1.get_score())/std::log(5), parent1.get_score());

	for(unsigned long long iteration = 1 ; iteration < ITERATION ; iteration++)
	{
		child.crossover(parent1, parent2);
		//child.mutation(parent1, var);
		if((tmp = child.count_score()) > std::min(parent1.get_score(), parent2.get_score())) 
		{
			if(parent1.get_score() > parent2.get_score()) std::swap(parent2, child);
			else std::swap(parent1, child);
			//std::swap(parent1, child);
			mutation_success++;
			std::printf("Iteration : %5llu : %lf\n", iteration, tmp);
		}
		// if(!(iteration % section))
		// {
		// 	if(mutation_success * 5 > section) var /= alpha;
		// 	else if(mutation_success * 5 < section) var *= alpha;
		// 	//var = std::max(0.0001, var);
		// 	mutation_success = 0;
		// }
		tmp = std::max(parent1.get_score(), parent2.get_score());
		std::fprintf(stderr, "Iteration : %5llu %lf %lf %lf\n", iteration, var, std::log(tmp)/std::log(5), tmp);
	}
	//if(parent1.get_score() > parent2.get_score()) parent1.print();
	//else parent2.print();
	return 0;
}