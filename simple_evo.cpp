#include <iostream>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/population.hpp>

using namespace pagmo;

// Returns the total number of weights for a given neural network structure
int get_size_of_net(const std::vector<int> &structure) {
    int total_size = 0;
    for (size_t i = 0; i < structure.size() - 1; ++i) {
        total_size += (structure[i] + 1) * structure[i + 1];
    }
    return total_size;
}

// Rescue problem that will run moos-ivp-learn simulations
struct rescue_problem {
    std::vector<int> m_structure;
    std::vector<std::vector<double>> m_action_bounds;
    int m_num_weights = 0;
    std::vector<double> m_lower_weight_bounds;
    std::vector<double> m_upper_weight_bounds;

    rescue_problem() : m_structure({}), m_action_bounds({}) {}
    rescue_problem(
        const std::vector<int> &structure, 
        const std::vector<std::vector<double>> &action_bounds
    ) : m_structure(structure), m_action_bounds(action_bounds) {
        m_num_weights = get_size_of_net(m_structure);
        m_lower_weight_bounds = std::vector<double>(m_num_weights, -1e19);
        m_upper_weight_bounds = std::vector<double>(m_num_weights, 1e19);
    }

    // Implementation of the objective function.
    vector_double fitness(const vector_double &dv) const
    {
        // return {dv[0] * dv[3] * (dv[0] + dv[1] + dv[2]) + dv[2]};
        return {0.0};
    }
    // Implementation of the box bounds.
    std::pair<vector_double, vector_double> get_bounds() const
    {
        // This is how pagmo knows the number of parameters
        // in the solution space.
        // If the length of these two vectors don't match, there is a problem.
        return {m_lower_weight_bounds, m_upper_weight_bounds};
    }
};

int main()
{
    pagmo::random_device::set_seed(42); // Set a fixed seed for deterministic results

    // 1 - Instantiate a pagmo problem constructing it from a UDP
    // (i.e., a user-defined problem, in this case the 30-dimensional
    // generalised Schwefel test function).
    std::vector<int> in_structure = {8,10,2};
    std::vector<std::vector<double>> in_action_bounds = {{-1.0, 1.0}};
    problem prob{rescue_problem{in_structure, in_action_bounds}};

    // Compute the value of the objective function
    int num_weights = get_size_of_net(in_structure);
    std::vector<double> example_weights(num_weights, 1.0);
    std::cout << "Value of the objfun in (1, 2, 3, 4): " << prob.fitness(example_weights)[0] << '\n';

    // Fetch the lower/upper bounds for the first variable.
    std::cout << "Lower bounds: [" << prob.get_lb()[0] << "]\n";
    std::cout << "Upper bounds: [" << prob.get_ub()[0] << "]\n\n";

    // Print p to screen.
    std::cout << prob << '\n';

    // // 2 - Instantiate a pagmo algorithm (self-adaptive differential
    // // evolution, 10,000 generations).
    // algorithm algo{sade(10000)};

    // // Create a population of 20 individuals for the problem.
    // population pop{prob, 20u};

    // // Evolve the population using the algorithm.
    // pop = algo.evolve(pop);

    // // Print the fitness of the best solution.
    // std::cout << pop.champion_f()[0] << '\n';
}
