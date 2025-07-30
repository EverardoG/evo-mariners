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
#include <csignal>
#include <sys/wait.h>  // for waitpid

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/population.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>

using namespace std;
using namespace pagmo;

static mutex cout_mtx;

// Global flag for graceful shutdown
volatile sig_atomic_t running = 1;

// Signal handler for SIGINT (Ctrl+C)
void signalHandler(int signal) {
    if (signal == SIGINT) {
        running = 0;
        std::cout << "\nReceived Ctrl+C, shutting down gracefully..." << std::endl;
    }
}

// For storing points
struct XYPoint {
    double x;
    double y;

    // Constructor with x,y values
    XYPoint(double x_val, double y_val) : x(x_val), y(y_val) {}
};

// And pose
struct XYPose : public XYPoint {
    double heading;

    // Constructor with x,y,heading values
    XYPose(double x_val, double y_val, double heading_val) 
        : XYPoint(x_val, y_val), heading(heading_val) {}
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
                double x_val = stod(x_str);
                double y_val = stod(y_str);
                points.push_back(XYPoint(x_val, y_val));
            } catch (const exception& e) {
                // There was a problem reading the file
                success = false;
                continue;
            }
        }
    }
    return {points, success};
}

bool write_swimmers_txt(const vector<XYPoint>& swimmer_pts, const string& filepath) {
    ofstream ofs(filepath, ios::out | ios::trunc);
    if (!ofs) {
        cout << "Error: cannot open " << filepath << " for writing\n";
        return false;
    }

    for (size_t i = 0; i < swimmer_pts.size(); ++i) {
        // Format: swimmer = name=p01, x=13, y=10
        ofs << "swimmer = name=p" 
            << setfill('0') << setw(2) << (i + 1) 
            << ", x=" << static_cast<double>(swimmer_pts[i].x)
            << ", y=" << static_cast<double>(swimmer_pts[i].y)
            << '\n';
    }

    ofs.flush();
    return true;
}

bool write_vpositions_txt(const vector<XYPose>& vehicle_poses, const string& filepath) {
    ofstream ofs(filepath, ios::out | ios::trunc);
    if (!ofs) {
        cout << "Error: cannot open " << filepath << " for writing\n";
        return false;
    }

    for (const auto& pose : vehicle_poses) {
        ofs << "x=" << pose.x 
            << ",y=" << pose.y 
            << ",heading=" << pose.heading 
            << '\n';
    }

    ofs.flush();
    return true;
}

int compute_swimmers_rescued(const vector<XYPoint>& vehicle_pts, const vector<XYPoint>& swimmer_pts, const double& rescue_rng_max = 5.0) {
    int total_swimmers = 0;
    
    // For each swimmer
    for (const auto& swimmer : swimmer_pts) {
        // Find closest vehicle point
        for (const auto& vehicle : vehicle_pts) {
            // Calculate Euclidean distance
            double dx = vehicle.x - swimmer.x;
            double dy = vehicle.y - swimmer.y;
            double distance = std::sqrt(dx*dx + dy*dy);

            // Add to the score if we rescued this swimmer
            if (distance < rescue_rng_max) {
                total_swimmers++;
                // Skip remaining vehicle points, go to next swimmer
                break;
            }
        }
    }
    
    return total_swimmers;
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
    int m_current_generation = 0; // Track the current generation

    rescue_problem() : m_structure({}), m_action_bounds({}) {}
    rescue_problem(
        const vector<int> &structure, 
        const vector<vector<double>> &action_bounds
    ) : m_structure(structure), m_action_bounds(action_bounds) {
        m_num_weights = get_size_of_net(m_structure);
        m_lower_weight_bounds = vector<double>(m_num_weights, -1e19);
        m_upper_weight_bounds = vector<double>(m_num_weights, 1e19);
    }

    void set_generation(int generation) {
        m_current_generation = generation;
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
        // Check if we should abort early
        if (!running) {
            return {100.0};
        }

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
        vector<XYPoint> swimmer_pts = {XYPoint(13.0, 10.0)};
        vector<XYPose> vehicle_poses = {XYPose(13.0,-10.0,181.0)};

        string host_swimmers_txt_dir = host_workdir+"swimmers.txt";
        string app_swimmers_txt_dir = apptainer_workdir+"swimmers.txt";
        if (!write_swimmers_txt(swimmer_pts, host_swimmers_txt_dir)) return {101.0};
        if (!write_vpositions_txt(vehicle_poses, host_workdir+"vpositions.txt")) return {102.0};
        if (!write_neural_network_csv(dv, host_workdir)) return {103.0};
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
            "--autodeploy",
            "--swim_file="+app_swimmers_txt_dir
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
            launch_cmd,
            process_node_reports_cmd,
            filter_node_reports_cmd
        };
        string apptainer_exec_cmd = exec_cmd + "\"" + join(apptainer_cmds, " && ") + "\"";
        string redirect_cmd = " > "+host_log_dir+"apptainer_out.log 2>&1";
        string apptainer_exec_w_redirect_cmd = apptainer_exec_cmd + redirect_cmd;
        // cout << "Apptainer command: " << apptainer_exec_w_redirect_cmd << endl;
        
        // Execute with timeout and signal handling
        pid_t child_pid = fork();
        if (child_pid == 0) {
            // Child process: execute the command
            execl("/bin/sh", "sh", "-c", apptainer_exec_w_redirect_cmd.c_str(), (char*)NULL);
            exit(1);  // If execl fails
        } else if (child_pid > 0) {
            // Parent process: wait for child with periodic checks
            int status;
            pid_t result;
            
            // Poll every second to check if we should terminate
            while ((result = waitpid(child_pid, &status, WNOHANG)) == 0) {
                if (!running) {
                    // Send SIGTERM first, then SIGKILL if needed
                    kill(child_pid, SIGTERM);
                    sleep(2);  // Give it 2 seconds to terminate gracefully
                    if (waitpid(child_pid, &status, WNOHANG) == 0) {
                        kill(child_pid, SIGKILL);
                        waitpid(child_pid, &status, 0);  // Clean up zombie
                    }
                    return {104.0};  // Return early
                }
                sleep(1);  // Check every second
            }
            
            if (result == -1) {
                // Error in waitpid
                return {105.0};
            }
        } else {
            // Fork failed
            return {106.0};
        }

        // Check again before processing results
        if (!running) {
            return {107.0};
        }

        //  5) Process the logs (or post-processed info) that were saved to get the fitness
        //  We should end up with something like team_positions.csv
        pair<vector<XYPoint>, bool> result = read_xy_csv("path/to/file.csv");
        vector<XYPoint> vehicle_pts = result.first;
        bool success = result.second;
        if (!success) return {108.0};

        int swimmers_rescued = compute_swimmers_rescued(vehicle_pts, swimmer_pts);
        double fitness_score = -static_cast<double>(swimmers_rescued);

        return {fitness_score};
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
    // Register signal handler
    std::signal(SIGINT, signalHandler);

    // seeding
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
        /*gen*/       50u,
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

    // cout << "Generation " << gen << "/" << max_generations << " - Best fitness: " << pop.champion_f()[0] << endl;

    // Evolve the population using the algorithm with graceful shutdown support
    const unsigned max_generations = 50u;
    for (unsigned gen = 1; gen <= max_generations && running; ++gen) { // Start at generation 1
        // Check if we should stop
        if (!running) {
            cout << "Evolution interrupted at generation " << gen << endl;
            break;
        }

        // Update the rescue problem
        rescue_problem updated_udp = rescue_problem(in_structure, in_action_bounds);
        updated_udp.set_generation(gen); // Update generation number
        problem updated_prob{updated_udp}; // Create updated problem

        // Create a new population with no individuals and the updated problem
        population updated_pop{updated_prob, 0};

        // Insert individuals from the old population into the new population
        const auto &decision_vectors = pop.get_x(); // Get all decision vectors
        const auto &fitness_vectors = pop.get_f(); // Get all fitness vectors
        for (size_t i = 0; i < pop.size(); ++i) {
            updated_pop.push_back(decision_vectors[i], fitness_vectors[i]); // Add both decision and fitness vectors
        }

        // Create a single-generation algorithm
        pagmo::sga single_gen_sga{
            1u,           // Only 1 generation
            0.9,          // crossover probability
            1.0,          // eta_c
            0.2,          // mutation probability
            param_m,      // mutation parameter
            3u,           // tournament size
            "exponential", // crossover type
            "gaussian",   // mutation type
            "tournament"  // selection type
        };
        algorithm single_gen_algo{single_gen_sga};
        
        // Evolve one generation
        pop = single_gen_algo.evolve(updated_pop);

        // Tell us the champion
        cout << "Generation " << gen << "/" << max_generations << " - Best fitness: " << pop.champion_f()[0] << endl;
    }

    // Print the fitness of the best solution.
    cout << "Final best fitness: " << pop.champion_f()[0] << '\n';
}
