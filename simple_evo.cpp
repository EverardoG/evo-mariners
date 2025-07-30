#include <iostream>
#include <filesystem>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <sstream>
#include <iomanip>  // for std::setprecision
#include <random>
#include <stdexcept>
#include <fstream>
#include <cstdlib>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/population.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>

static std::mutex cout_mtx;

using namespace pagmo;

// Returns the absolute directory of this cpp file
std::filesystem::path get_source_dir() {
    return std::filesystem::canonical(std::filesystem::path(__FILE__)).parent_path();
}

// Returns the total number of weights for a given neural network structure
int get_size_of_net(const std::vector<int> &structure) {
    int total_size = 0;
    for (size_t i = 0; i < structure.size() - 1; ++i) {
        total_size += (structure[i] + 1) * structure[i + 1];
    }
    return total_size;
}

/// Convert a vector of doubles into a CSV line (no newline appended).
/// @param row      The vector of values to serialize.
/// @param precision  Number of digits after the decimal point (default: 6).
/// @return a string like "1.234567,2.000000,3.141593"
std::string vec_double_to_str(const std::vector<double> &row, int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision);
    for (size_t i = 0; i < row.size(); ++i) {
        if (i) {
            oss << ',';          // comma delimiter
        }
        oss << row[i];
    }
    return oss.str();
}

/// Convert a vector of ints into a CSV line (no newline appended).
/// @param row  The vector of integer values to serialize.
/// @return     A string like "1,2,3,4"
std::string vec_int_to_str(const std::vector<int> &row) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < row.size(); ++i) {
        if (i) {
            oss << ',';  // comma delimiter
        }
        oss << row[i];
    }
    return oss.str();
}

/// Generate an initial Pagmo population whose decision vectors are sampled
/// uniformly in [low, high] (per component), regardless of the UDP’s full bounds.
/// @param prob      The Pagmo problem (wraps your UDP).
/// @param pop_size  Number of individuals to generate.
/// @param low       Minimum value for each gene (default -1.0).
/// @param high      Maximum value for each gene (default +1.0).
/// @param seed      RNG seed (default: random_device).
/// @returns         A Pagmo population of size pop_size.
pagmo::population generate_initial_population(
    const pagmo::problem &prob,
    std::size_t pop_size,
    double low = -1.0,
    double high = +1.0,
    unsigned seed = std::random_device{}()
) {
    // RNG & distribution for sampling in [low, high]
    std::mt19937_64 rng{seed};
    std::uniform_real_distribution<double> dist{low, high};

    // Problem dimension
    const std::size_t dim = prob.get_nx();

    // Start with an empty population
    pagmo::population pop{prob, 0u};

    // Sample and push each individual
    for (std::size_t i = 0; i < pop_size; ++i) {
        pagmo::vector_double x(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            x[d] = dist(rng);
        }
        pop.push_back(x);
    }

    return pop;
}

/// Compute the SGA Gaussian‐mutation parameter m_param_m required
/// to get a per‐gene mutation standard deviation of `desired_step`,
/// given your variable’s [lower, upper] bounds.
/// @throws std::invalid_argument if upper <= lower
inline double compute_sga_gaussian_param(double lower,
                                         double upper,
                                         double desired_step) {
    if (upper <= lower) {
        throw std::invalid_argument{"compute_sga_gaussian_param: upper must exceed lower"};
    }
    return desired_step / (upper - lower);
}

/// Join a list of strings with a given delimiter.
std::string join(const std::vector<std::string> &parts,
                 const std::string &delim) {
    std::ostringstream oss;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) oss << delim;
        oss << parts[i];
    }
    return oss.str();
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

    bool write_neural_network_csv(const vector_double &dv, std::string dir) const {
        std::string weights_str = vec_double_to_str(dv, 3);
        std::string structure_str = vec_int_to_str(m_structure);
        std::string action_bounds_str = vec_double_to_str(m_action_bounds[0], 3) + "," + vec_double_to_str(m_action_bounds[1], 3);

        // std::cout << weights_str << std::endl;
        // std::cout << structure_str << std::endl;
        // std::cout << action_bounds_str << std::endl;

        std::string neural_network_dir = dir + "/neural_network_config.csv";
        std::filesystem::path csv_file = neural_network_dir;

        // return {0.0};

        std::ofstream ofs(csv_file, std::ios::out | std::ios::trunc);
        if (!ofs) {
            std::cerr << "Error: cannot open " << csv_file << " for writing\n";
            return false;
        }

        // 4) Write each string as its own row
        ofs << weights_str << '\n';
        ofs << structure_str << '\n';
        ofs << action_bounds_str << '\n';

        // 5) (Optional) flush to ensure data is on disk immediately
        ofs.flush();
        return true;
    }

    // We will likely need a method for re-configuring the rescue_problem
    // in between generations when we have stochastic elements in our
    // problem - like randomly spawning swimmers
    // void set_configuration();

    // Implementation of the objective function.
    vector_double fitness(const vector_double &dv) const
    {
        // Looking into multi-threading
        std::lock_guard<std::mutex> lock(cout_mtx);
        std::cout << "fitness() on thread " << std::this_thread::get_id() << "\n";

        //  1) Determine what directory we will be working in
        //  Likely something like $HOME/hpc-share/tmp/slurm-<job-id>/process-<process-id>/thread-<thread-id>/
        std::string host_home = std::getenv("HOME");
        std::cout << host_home << std::endl;

        const char *sidchar = std::getenv("SLURM_JOB_ID");
        std::string sid = sidchar ? sidchar : "none";   

        std::thread::id thread_id = std::this_thread::get_id();
        std::ostringstream oss;
        oss << thread_id;
        std::string tid = oss.str();

        pid_t process_id = getpid();
        std::string pid = std::to_string(process_id);

        std::string host_workdir = host_home+"/hpc-share/tmp/slurm-"+sid+"/process-"+pid+"/thread-"+tid+"/";
        std::string apptainer_workdir = "/home/moos/hpc-share/tmp/slurm-"+sid+"/process-"+pid+"/thread-"+tid+"/";
        std::cout << "host_workdir: " << host_workdir << std::endl;
        std::filesystem::create_directories(host_workdir);

        std::filesystem::path source_dir = get_source_dir();
        std::filesystem::path apptainer_path = source_dir / "apptainer" / "ubuntu_20.04_ivp_2680_learn.sif";

        std::cout << "source dir: " << get_source_dir() << std::endl;
    
        //  2) Set up the directory
        //  Write out neural network csv parameters to a csv file in that directory
        if (!write_neural_network_csv(dv, host_workdir)) return {0.0};

        //  3) Run the apptainer instance.
        // Build launch command for moos
        std::vector<std::string> launch_args = {
            "10", // timewarp
            "--xlaunched",
            "--logdir="+apptainer_workdir+"logs/",
            "--trim",
            "--neural_network_dir="+apptainer_workdir+"neural_networks/",
            "--uMayFinish",
            "--nogui",
            "--rescuebehavior=NeuralNetwork"
        };
        std::string launch_cmd = std::string{"./launch.sh "} + join(launch_args, " ");
        // std::cout << "launch: " << launch_cmd << std::endl;

        // Build apptainer exec command for launching mission
        // std::string exec = std::"apptainer exec apptainer/ubuntu_20.04_ivp_2680_learn.sif /bin/bash -c"
        std::vector<std::string> exec_pieces = {
            "apptainer exec",
            "--cleanenv",
            "--containall",
            "--contain",
            "--net",
            "--network=fakeroot",
            "--fakeroot",
            "--bind "+host_home+"/hpc-share:/home/moos/hpc-share",
            "--writable-tmpfs",
            apptainer_path.string(),
            "/bin/bash -c "
        };
        std::string exec_cmd = join(exec_pieces, " ");
        std::vector<std::string> apptainer_cmds = {
            "cd /home/moos/moos-ivp-learn/missions/alpha_learn",
            "echo \'x=13.0,y=-10.0,heading=181\' > vpositions.txt",
            launch_cmd
        };
        std::string apptainer_exec_cmd = exec_cmd + "\"" + join(apptainer_cmds, " && ") + "\"";
        std::cout << "app: " << apptainer_exec_cmd << std::endl;
        std::system(apptainer_exec_cmd.c_str());

        //  3a) Launch Mission - IN PROGRESS
        //  3b) Auto-deploy when ready - DONE
        //  3c) Hit end condition (all swimmers rescued, vehicle out of bounds, or timeout)
        //  uMayFinish. pMissionMonitor

        // std::system(<apptainer command goes in here>)

        // apptainer exec ....
        //   inside the exec: ./launch.sh
        //      inside ./launch.sh: runs pAntler, runs uMayFinish. 
        // Note: Logs are saved in a shared directory (still accessible after apptainer exits)

        //  3d) Stop running moos - taken care of by uMayFinish
        //  3e) Run post-processing scripts on logs (process_node_reports, filter_duplicate_rows)

        // apptainer exec ...
        //   runs post processing scripts

        //  3f) Kill the container - Not necessary because apptainer exec automically kills instance

        //  4) Recieve some kind of kill command or stop command from apptainer
        //  Maybe apptainer just exits on its own, and that's how cpp
        //  knows to keep going
        //  Not necessary because apptainer automatically stops after running exec

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
    unsigned int seed = 42u;

    pagmo::random_device::set_seed(seed); // Set a fixed seed for deterministic results

    // 1 - Instantiate a pagmo problem constructing it from a UDP
    // (i.e., a user-defined problem, in this case the 30-dimensional
    // generalised Schwefel test function).
    std::vector<int> in_structure = {8,10,2};
    std::vector<std::vector<double>> in_action_bounds = {{0.0, 1.0}, {-180.0, 180.0}};
    problem prob{rescue_problem{in_structure, in_action_bounds}};

    // std::cout << "Pagmo will use up to "
    //     << std::thread::hardware_concurrency()
    //     << " threads.\n";

    // Compute the value of the objective function
    int num_weights = get_size_of_net(in_structure);
    std::vector<double> example_weights(num_weights, 1.0);
    // std::cout << "Value of the objfun in (1, 2, 3, 4): " << prob.fitness(example_weights)[0] << '\n';

    // Print p to screen.
    // std::cout << prob << '\n';

    // Configure SGA
    //    - 200 generations
    //    - 90% crossover (exponential by default)
    //    - 20% per‑gene mutation
    //    - uniform mutation (redraw within [lb,ub])
    //    - tournament size 3
    std::pair<pagmo::vector_double, pagmo::vector_double> bounds = prob.get_bounds();
    const pagmo::vector_double &lower = bounds.first;
    const pagmo::vector_double &upper = bounds.second;
    double desired_sigma = 1.0;  // mutation step
    double param_m = compute_sga_gaussian_param(lower[0], upper[0], desired_sigma);
    pagmo::sga sga_setup{
        /*gen*/       1u,
        /*cr*/        0.9,
        /*eta_c*/     1.0,      // SBX index (unused here)
        /*m*/         0.2,      // 20% mutation rate
        /*param_m*/   param_m,      // mutation step
        /*param_s*/   3u,       // tournament size
        /*xover*/     "exponential",
        /*mutation*/  "gaussian",
        /*sel*/       "tournament"
    };

    // 2 - Instantiate a pagmo algorithm
    algorithm algo{sga_setup};

    // Create a population of 50 individuals for the problem.
    population pop = generate_initial_population(prob, 10, -1.0, +1.0, seed);

    // Evolve the population using the algorithm.
    pop = algo.evolve(pop);

    // Print the fitness of the best solution.
    std::cout << pop.champion_f()[0] << '\n';
}
