#include <iostream>
#include <filesystem>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <sstream>
#include <iomanip>  // for setprecision
#include <random>
#include <stdexcept>
#include <fstream>
#include <cstdlib>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/population.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>

using namespace std;
using namespace pagmo;

static mutex cout_mtx;

// For storing points
struct XYPoint {
    double x;
    double y;

    // Constructor with x,y values
    XYPoint(double x_val, double y_val) : x(x_val), y(y_val) {}
};

// Function to read CSV and return vector of points
pair<vector<XYPoint>, bool> read_xy_csv(const string& filepath) {
    vector<XYPoint> points;
    bool success = true;
    ifstream file(filepath);
    
    if (!file.is_open()) {
        return {points, false};
    }

    string line;
    // Skip header line
    getline(file, line);
    
    // Read data lines
    while (getline(file, line)) {
        stringstream ss(line);
        string x_str, y_str;
        
        // Split line by comma
        if (getline(ss, x_str, ',') && getline(ss, y_str, ',')) {
            try {
                XYPoint point;
                point.x = stod(x_str);
                point.y = stod(y_str);
                points.push_back(point);
            } catch (const exception& e) {
                // There was a problem reading the file
                success = false;
                continue;
            }
        }
    }
    return {points, success};
}

// Returns the absolute directory of this cpp file
filesystem::path get_source_dir() {
    return filesystem::canonical(filesystem::path(__FILE__)).parent_path();
}

// Returns the total number of weights for a given neural network structure
int get_size_of_net(const vector<int> &structure) {
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
string vec_double_to_str(const vector<double> &row, int precision = 6) {
    ostringstream oss;
    oss << fixed << setprecision(precision);
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
string vec_int_to_str(const vector<int> &row) {
    ostringstream oss;
    for (size_t i = 0; i < row.size(); ++i) {
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
    size_t pop_size,
    double low = -1.0,
    double high = +1.0,
    unsigned seed = std::random_device{}()
) {
    // RNG & distribution for sampling in [low, high]
    mt19937_64 rng{seed};
    uniform_real_distribution<double> dist{low, high};

    // Problem dimension
    const size_t dim = prob.get_nx();

    // Start with an empty population
    pagmo::population pop{prob, 0u};

    // Sample and push each individual
    for (size_t i = 0; i < pop_size; ++i) {
        pagmo::vector_double x(dim);
        for (size_t d = 0; d < dim; ++d) {
            x[d] = dist(rng);
        }
        pop.push_back(x);
    }

    return pop;
}

/// Compute the SGA Gaussian‐mutation parameter m_param_m required
/// to get a per‐gene mutation standard deviation of `desired_step`,
/// given your variable’s [lower, upper] bounds.
/// @throws invalid_argument if upper <= lower
inline double compute_sga_gaussian_param(double lower,
                                         double upper,
                                         double desired_step) {
    if (upper <= lower) {
        throw invalid_argument{"compute_sga_gaussian_param: upper must exceed lower"};
    }
    return desired_step / (upper - lower);
}

/// Join a list of strings with a given delimiter.
string join(const vector<string> &parts,
                 const string &delim) {
    ostringstream oss;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) oss << delim;
        oss << parts[i];
    }
    return oss.str();
}

// Rescue problem that will run moos-ivp-learn simulations
struct rescue_problem {
    vector<int> m_structure;
    vector<vector<double>> m_action_bounds;
    int m_num_weights = 0;
    vector<double> m_lower_weight_bounds;
    vector<double> m_upper_weight_bounds;

    rescue_problem() : m_structure({}), m_action_bounds({}) {}
    rescue_problem(
        const vector<int> &structure, 
        const vector<vector<double>> &action_bounds
    ) : m_structure(structure), m_action_bounds(action_bounds) {
        m_num_weights = get_size_of_net(m_structure);
        m_lower_weight_bounds = vector<double>(m_num_weights, -1e19);
        m_upper_weight_bounds = vector<double>(m_num_weights, 1e19);
    }

    bool write_neural_network_csv(const vector_double &dv, string dir) const {
        string weights_str = vec_double_to_str(dv, 3);
        string structure_str = vec_int_to_str(m_structure);
        string action_bounds_str = vec_double_to_str(m_action_bounds[0], 3) + "," + vec_double_to_str(m_action_bounds[1], 3);

        // cout << weights_str << endl;
        // cout << structure_str << endl;
        // cout << action_bounds_str << endl;

        string neural_network_dir = dir + "/neural_network_config.csv";
        filesystem::path csv_file = neural_network_dir;

        // return {0.0};

        ofstream ofs(csv_file, ios::out | ios::trunc);
        if (!ofs) {
            cerr << "Error: cannot open " << csv_file << " for writing\n";
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
        lock_guard<mutex> lock(cout_mtx);
        // cout << "fitness() on thread " << this_thread::get_id() << "\n";

        //  1) Determine what directory we will be working in
        //  Likely something like $HOME/hpc-share/tmp/slurm-<job-id>/process-<process-id>/thread-<thread-id>/

        const char *sidchar = getenv("SLURM_JOB_ID");
        string sid = sidchar ? sidchar : "none";   

        thread::id thread_id = this_thread::get_id();
        ostringstream oss;
        oss << thread_id;
        string tid = oss.str();

        pid_t process_id = getpid();
        string pid = to_string(process_id);

        string host_home = getenv("HOME");
        // cout << host_home << endl;
        string host_workdir = host_home+"/hpc-share/tmp/slurm-"+sid+"/process-"+pid+"/thread-"+tid+"/";
        string host_log_dir = host_workdir+"logs/";

        string apptainer_workdir = "/home/moos/hpc-share/tmp/slurm-"+sid+"/process-"+pid+"/thread-"+tid+"/";
        string apptainer_log_dir = apptainer_workdir+"logs/";

        // cout << "host_workdir: " << host_workdir << endl;
        filesystem::create_directories(host_log_dir);

        filesystem::path source_dir = get_source_dir();
        filesystem::path apptainer_path = source_dir / "apptainer" / "ubuntu_20.04_ivp_2680_learn.sif";

        // cout << "source dir: " << get_source_dir() << endl;
    
        //  2) Set up the directory
        //  Write out neural network csv parameters to a csv file in that directory
        if (!write_neural_network_csv(dv, host_workdir)) return {0.0};
        vector<XYPoint> swimmer_pts = {XYPoint(13.0, 10.0)};

        //  3) Run the apptainer instance.
        // Build launch command for moos

        vector<string> launch_args = {
            "10", // timewarp
            "--xlaunched",
            "--logdir="+apptainer_log_dir,
            "--trim",
            "--neural_network_dir="+apptainer_workdir+"neural_networks/",
            "--uMayFinish",
            "--nogui",
            "--rescuebehavior=NeuralNetwork",
            "--autodeploy"
        };
        string launch_cmd = string{"./launch.sh "} + join(launch_args, " ");
        // cout << "launch: " << launch_cmd << endl;

        // Build apptainer exec command for launching mission
        // string exec = "apptainer exec apptainer/ubuntu_20.04_ivp_2680_learn.sif /bin/bash -c"
        vector<string> exec_pieces = {
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
        string exec_cmd = join(exec_pieces, " ");
        string process_node_reports_cmd = "process_node_reports "+\
            apptainer_log_dir+"XLOG_SHORESIDE/XLOG_SHORESIDE.alog "+\
            apptainer_log_dir+"abe_positions.csv";
        string filter_node_reports_cmd = "csv_filter_duplicate_rows "+\
            apptainer_log_dir+"abe_positions.csv "+\
            apptainer_log_dir+"abe_positions_filtered.csv";
        vector<string> apptainer_cmds = {
            "cd /home/moos/moos-ivp-learn/missions/alpha_learn",
            "echo \'x=13.0,y=-10.0,heading=181\' > vpositions.txt",
            launch_cmd,
            process_node_reports_cmd,
            filter_node_reports_cmd
        };
        string apptainer_exec_cmd = exec_cmd + "\"" + join(apptainer_cmds, " && ") + "\"";
        string redirect_cmd = " > "+host_log_dir+"apptainer_out.log 2>&1";
        string apptainer_exec_w_redirect_cmd = apptainer_exec_cmd + redirect_cmd;
        // cout << "Apptainer command: " << apptainer_exec_w_redirect_cmd << endl;
        system(apptainer_exec_w_redirect_cmd.c_str());

        //  3a) Launch Mission - IN PROGRESS
        //  3b) Auto-deploy when ready - DONE
        //  3c) Hit end condition (all swimmers rescued, vehicle out of bounds, or timeout)
        //  uMayFinish. pMissionMonitor

        // system(<apptainer command goes in here>)

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
        pair<vector<XYPoint>, bool> result = read_xy_csv("path/to/file.csv");
        vector<XYPoint> vehicle_pts = result.first;
        bool success = result.second;
        if (!success) {
            cout << "Failed to read points from file\n";
        }


        return {0.0};
    }
    // Implementation of the box bounds.
    pair<vector_double, vector_double> get_bounds() const
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
    vector<int> in_structure = {8,10,2};
    vector<vector<double>> in_action_bounds = {{0.0, 1.0}, {-180.0, 180.0}};
    problem prob{rescue_problem{in_structure, in_action_bounds}};

    // cout << "Pagmo will use up to "
    //     << thread::hardware_concurrency()
    //     << " threads.\n";

    // Compute the value of the objective function
    int num_weights = get_size_of_net(in_structure);
    vector<double> example_weights(num_weights, 1.0);
    // cout << "Value of the objfun in (1, 2, 3, 4): " << prob.fitness(example_weights)[0] << '\n';

    // Print p to screen.
    // cout << prob << '\n';

    // Configure SGA
    //    - 200 generations
    //    - 90% crossover (exponential by default)
    //    - 20% per‑gene mutation
    //    - uniform mutation (redraw within [lb,ub])
    //    - tournament size 3
    pair<pagmo::vector_double, pagmo::vector_double> bounds = prob.get_bounds();
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
    cout << pop.champion_f()[0] << '\n';
}
