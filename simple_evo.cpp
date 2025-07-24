#include <iostream>
#include <filesystem>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <sstream>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/population.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>

static std::mutex cout_mtx;

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
        // Looking into multi-threading
        std::lock_guard<std::mutex> lock(cout_mtx);
        std::cout << "fitness() on thread " << std::this_thread::get_id() << "\n";

        //  1) Determine what directory we will be working in
        //  Likely something like $HOME/hpc-share/tmp/slurm-<job-id>/process-<process-id>/thread-<thread-id>/
        std::string home = std::getenv("HOME");

        const char *sidchar = std::getenv("SLURM_JOB_ID");
        std::string sid = sidchar ? sidchar : "none";   

        std::thread::id thread_id = std::this_thread::get_id();
        std::ostringstream oss;
        oss << thread_id;
        std::string tid = oss.str();

        pid_t process_id = getpid();
        std::string pid = std::to_string(process_id);

        std::string dir = home + "/hpc-share/tmp/slurm-"+sid+"/process-"+pid+"/thread-"+tid;
        std::filesystem::create_directories(dir);
    
        //  2) Set up the directory
        //  Write out neural network cs parameters to a csv file in that directory
        //  return {dv[0] * dv[3] * (dv[0] + dv[1] + dv[2]) + dv[2]};

        //  3) Run the apptainer instance. Put the csv directory as a parameter
        //  Put the output log directory as a parameter

        //  3a) Launch Mission
        //  3b) Auto-deploy when ready
        //  3c) Hit end condition (all swimmers rescued, vehicle out of bounds, or timeout)
        //  3d) Stop running moos
        //  3e) Run post-processing scripts on logs (process_node_reports, filter_duplicate_rows)
        //  3f) Kill the container

        //  4) Recieve some kind of kill command or stop command from apptainer
        //  Maybe apptainer just exits on its own, and that's how cpp
        //  knows to keep going

        //  5) Process the logs (or post-processed info) that were saved to get the fitness
        //  We should end up with something like team_positions.csv
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

    // Tell pagmo this is thread safe so it runs parallel evaluations
    pagmo::thread_safety get_thread_safety() const noexcept {
        return pagmo::thread_safety::constant;
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

    std::cout << "Pagmo will use up to "
        << std::thread::hardware_concurrency()
        << " threads.\n";

    // Compute the value of the objective function
    int num_weights = get_size_of_net(in_structure);
    std::vector<double> example_weights(num_weights, 1.0);
    std::cout << "Value of the objfun in (1, 2, 3, 4): " << prob.fitness(example_weights)[0] << '\n';

    // Print p to screen.
    std::cout << prob << '\n';

    // 2 - Instantiate a pagmo algorithm
    algorithm algo{sade(1)};

    // Create a population of 20 individuals for the problem.
    population pop{prob, 20u};

    // Evolve the population using the algorithm.
    pop = algo.evolve(pop);

    // Print the fitness of the best solution.
    std::cout << pop.champion_f()[0] << '\n';
}
