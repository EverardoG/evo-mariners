#include <iostream>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/population.hpp>

using namespace pagmo;

int main()
{
    // 1 - Instantiate a pagmo problem constructing it from a UDP
    // (i.e., a user-defined problem, in this case the 30-dimensional
    // generalised Schwefel test function).
    problem prob{schwefel(30)};

    // 2 - Instantiate a pagmo algorithm (self-adaptive differential
    // evolution, 10,000 generations).
    algorithm algo{sade(10000)};

    // Create a population of 20 individuals for the problem.
    population pop{prob, 20u};

    // Evolve the population using the algorithm.
    pop = algo.evolve(pop);

    // Print the fitness of the best solution.
    std::cout << pop.champion_f()[0] << '\n';
}
