#include <cstdio>
#include "gene.cpp"

int main(int N, char ** args)
{
	if(N != 3) {printf("Wrong format.\n"); exit(-1);}
	GENE parent, child(false);
	double var, alpha = atof(args[2]);
	unsigned mutation_success = 0;
	unsigned long long iteration = 0, section = std::atoll(args[1]);
	while(1)
	{
		iteration++;
		child.mutation(parent, var);
		if(child.score > parent.score) 
		{
			std::swap(parent, child);
			mutation_success++;

		}
		if(!(iteration % section))
		{
			if(mutation_success * 5 > section) var /= alpha;
			else if(mutation_success * 5 < section) var *= alpha;
			mutation_success++;
		}
	}
	return 0;
}