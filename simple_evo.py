import os
from pathlib import Path
import random
from copy import deepcopy
from typing import List
import multiprocessing
import subprocess
import csv
import math
import argparse

from tqdm import tqdm
import pandas as pd
from shapely.geometry import Point

class Pose():
    def __init__(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading

class Individual():
    def __init__(self, weights, temp_id):
        self.weights = weights
        self.temp_id = temp_id
        self.rollout_fitnesses = []
        self.fitness = None

class IndividualEvalIn():
    def __init__(self, individual, seed, rollout_id):
        self.individual = individual
        self.seed = seed
        self.rollout_id = rollout_id
        self.fitness = None

class IndividualEvalOut():
    def __init__(self, fitness):
        self.fitness = fitness

class IndividualSummary():
    def __init__(self, individual, seeds, eval_outs):
        self.individual = individual
        self.seeds = seeds
        self.eval_outs = eval_outs

def getSizeOfNet(structure: List[int]) -> int:
    total_size = 0
    for i in range(len(structure)-1):
        total_size += (structure[i] + 1)*structure[i+1]
    return total_size

def getRandomWeights(num_weights: int) -> List[float]:
    return [2*random.random()-1 for _ in range(num_weights)]

def writeSwimmersTxt(swimmer_pts, filepath):
    """
    Write swimmers data to a text file.

    Args:
        swimmer_pts: List of shapely Point objects representing swimmer positions
        filepath: pathlib.Path object for the output file

    Raises:
        IOError: If the file cannot be opened or written to
        Exception: For any other errors during file operations
    """
    with open(filepath, 'w') as f:
        # Write the top line
        f.write("poly = pts={60,10:-75.5402,-54.2561:-36.9866,-135.58:98.5536,-71.3241}\n")

        # Write swimmer data
        for i, swimmer_pt in enumerate(swimmer_pts):
            # Format: swimmer = name=p01, x=13, y=10
            f.write(f"swimmer = name=p{i+1:02d}, x={swimmer_pt.x}, y={swimmer_pt.y}\n")

def writeVpositionsTxt(vehicle_poses, filepath):
    """
    Write vehicle positions data to a text file.

    Args:
        vehicle_poses: List of Pose objects with x, y, and heading attributes
        filepath: pathlib.Path object for the output file

    Raises:
        IOError: If the file cannot be opened or written to
        Exception: For any other errors during file operations
    """
    with open(filepath, 'w') as f:
        for pose in vehicle_poses:
            f.write(f"x={pose.x},y={pose.y},heading={pose.heading}\n")

def writeNeuralNetworkCsv(weights, structure, action_bounds, filepath):
    """
    Write neural network parameters to a CSV file.

    Args:
        weights: List of weight values (floats)
        structure: List of integers representing the network structure
        action_bounds: List of two lists - [lower_bounds, upper_bounds]
        filepath: pathlib.Path object for the output file

    Raises:
        IOError: If the file cannot be opened or written to
        Exception: For any other errors during file operations
    """
    # Convert data to strings with proper formatting
    weights_str = ','.join(f"{w:.3f}" for w in weights)
    structure_str = ','.join(str(s) for s in structure)
    action_bounds_str = ','.join(f"{b:.3f}" for b in action_bounds[0]) + ',' + \
                       ','.join(f"{b:.3f}" for b in action_bounds[1])

    with open(filepath, 'w') as f:
        f.write(f"{weights_str}\n")
        f.write(f"{structure_str}\n")
        f.write(f"{action_bounds_str}\n")

def readXyCsv(filepath):
    """
    Reads a CSV file and returns a list of Point objects.
    Assumes the first line is a header and each subsequent line is 'x,y'.
    Raises an exception if the file cannot be read.
    """
    points = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # Skip header
        for row in reader:
            x_val = float(row[0])
            y_val = float(row[1])
            points.append(Point(x_val, y_val))
    return points

def getSourceDir():
    """
    Returns the absolute directory of this python file.

    Returns:
        pathlib.Path: The directory containing the current Python script
    """
    return Path(__file__).parent.resolve()

class EvolutionaryAlgorithm():
    def __init__(self, config_dir: str):
        self.use_multiprocessing = False
        self.num_processes = 10

        self.num_trials = 1
        self.num_generations = 100
        self.gen = 0
        self.trial_id = None

        self.config_seed = 0
        self.random_seed_val= None
        self.increment_seed_every_trial = True

        self.neural_network_structure = [8, 10, 5, 2]
        self.neural_network_size = getSizeOfNet(self.neural_network_structure)
        self.neural_network_action_bounds = [[0.0, 1.0], [-180.0, 180.0]]

        self.population_size = 50

        self.num_rollouts_per_indivdiual = 1
        self.rpi = self.num_rollouts_per_indivdiual

        self.n_elites = 5
        self.tournament_size = 3

        self.mut_indpb = 0.2
        self.mut_std = 1.0

        # self.swimmer_generation = "fixed"
        self.default_swimmer_pts = [Point(12.0, -60.0)]
        self.default_vehicle_poses = [Pose(13.0, -20.0, 181.0)]

        self.moos_timewarp = 10

        self.host_root_folder = \
            Path(config_dir)
        self.host_root_folder.mkdir(parents=True, exist_ok=True)
        self.app_home = Path('/home/moos')
        self.app_root_folder = self.app_home / 'hpc-share'
        self.fitness_csv_file = None

        self.setupMapping()

    # This makes it possible to pass evaluation to multiprocessing
    # Without this, the pool tries to pickle the entire object, including itself
    # which it cannot do
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        del self_dict['map']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def writeNeuralNetworkCsv(self, individual, filepath):
        return writeNeuralNetworkCsv(
            individual.weights,
            self.neural_network_structure,
            self.neural_network_action_bounds,
            filepath
        )

    def computeSwimmersRescued(self, vehicle_pts, swimmer_pts):
        """
        Counts the number of swimmers rescued by vehicles.
        """
        rescue_rng_max = 5.0
        total_swimmers = 0
        for swimmer in swimmer_pts:
            for vehicle in vehicle_pts:
                dx = vehicle.x - swimmer.x
                dy = vehicle.y - swimmer.y
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < rescue_rng_max:
                    total_swimmers += 1
                    break
        return total_swimmers

    def setupMapping(self):
        if self.use_multiprocessing:
            self.pool = multiprocessing.Pool(processes=self.num_processes)
            self.map = self.pool.map_async
        else:
            self.map = map

    def resetSeed(self):
        self.random_seed_val= self.config_seed

    def generateIndividual(self, temp_id):
        return Individual(getRandomWeights(self.neural_network_size), temp_id)

    def population(self):
        return [self.generateIndividual(temp_id) for temp_id in range(self.population_size)]

    def getSeed(self):
        # Returns current seed. Increments by 1
        if self.config_seed is None:
            return None
        out = self.random_seed_val
        self.random_seed_val+= 1
        return out

    def wrapIndividuals(self, population, eval_summaries):
        for individual, eval_summary in zip(population, eval_summaries):
            individual.rollout_fitnesses = [eval_out.fitness for eval_out in eval_summary.eval_outs]
            individual.fitness = sum(individual.rollout_fitnesses) / len(individual.rollout_fitnesses)

    def buildIndividualSummaries(self, eval_ins, eval_outs):
        individual_summaries = []
        for i in range(int(len(eval_outs)/self.rpi)):
            individual_summaries.append(
                IndividualSummary(
                    individual=eval_ins[i*self.rpi].individual,
                    seeds=[eval_in.seed for eval_in in eval_ins[i*self.rpi:(i+1)*self.rpi]],
                    eval_outs=eval_outs[i*self.rpi:(i+1)*self.rpi]
                )
            )
        return individual_summaries

    def mutateIndividual(self, individual):
        for i in range(len(individual.weights)):
            if random.random() < self.mut_indpb:
                individual.weights[i] += random.gauss(0, self.mut_std)

    def selectAndMutate(self, population, individual_summaries):
        # Organize individuals by fitness
        population.sort(key=lambda individual: individual.fitness, reverse=True)

        # Take the elites
        offspring = [deepcopy(individual) for individual in population[:self.n_elites]]

        # Do a binary tournament selection to fill out the rest
        # Mutate the winner of the tournament
        while len(offspring) < self.population_size:
            competitors = random.sample(population, self.tournament_size)
            winner = deepcopy(max(competitors, key=lambda individual: individual.fitness))
            self.mutateIndividual(winner)
            offspring.append(winner)

        return offspring

    def evaluateIndividual(self, individual_eval_in):
        individual_folder = 'ind_'+str(individual_eval_in.individual.temp_id)
        rollout_folder = 'rollout_'+str(individual_eval_in.rollout_id)

        host_work_folder = self.host_root_folder / self.trial_folder / self.gen_folder / individual_folder / rollout_folder
        host_log_folder = host_work_folder / 'logs'
        host_log_folder.mkdir(parents=True, exist_ok=True)

        host_swimmers_txt_file = host_work_folder / 'swimmers.txt'
        host_vpositions_txt_file = host_work_folder / 'vpositions.txt'
        host_neural_net_csv_file = host_work_folder / 'neural_network_abe.csv'

        app_work_folder = self.app_root_folder / self.trial_folder / self.gen_folder / individual_folder / rollout_folder
        app_log_folder = app_work_folder / 'logs'
        apptainer_sif_file = getSourceDir() / "apptainer" / "ubuntu_20.04_ivp_2680_learn.sif";

        app_swimmers_txt_file = app_work_folder / 'swimmers.txt'
        app_vpositions_txt_file = app_work_folder / 'vpositions.txt'
        app_neural_net_csv_file = app_work_folder / 'neural_network_abe.csv'

        writeSwimmersTxt(self.default_swimmer_pts, host_swimmers_txt_file)
        writeVpositionsTxt(self.default_vehicle_poses, host_vpositions_txt_file)
        self.writeNeuralNetworkCsv(individual_eval_in.individual, host_neural_net_csv_file)

        # Build launch arguments
        launch_args = [
            str(self.moos_timewarp),  # timewarp
            "--xlaunched",
            f"--logdir={app_log_folder}/",
            f"--neural_network_dir={app_work_folder}/",
            "--uMayFinish",
            "--nogui",
            "--rescuebehavior=NeuralNetwork",
            "--autodeploy",
            f"--swim_file={app_swimmers_txt_file}",
            f"--vpositions={app_work_folder}/vpositions.txt",
            "--nostamp"
        ]
        launch_cmd = (
            "cd /home/moos/moos-ivp-learn/missions/alpha_learn && ./launch.sh " +
            " ".join(launch_args)
        )

        # Build apptainer exec command
        exec_pieces = [
            "apptainer exec",
            "--cleanenv",
            "--containall",
            "--contain",
            "--net",
            "--network=none",
            "--fakeroot",
            f"--bind {self.host_root_folder}:{self.app_root_folder}",
            "--writable-tmpfs",
            str(apptainer_sif_file),
            "/bin/bash -c "
        ]
        exec_cmd = " ".join(exec_pieces)

        process_node_reports_cmd = (
            f"process_node_reports {app_log_folder}/XLOG_SHORESIDE/XLOG_SHORESIDE.alog {app_log_folder}/"
        )
        filter_node_reports_cmd = (
            f"csv_filter_duplicate_rows {app_log_folder}/abe_positions.csv {app_log_folder}/abe_positions_filtered.csv"
        )
        apptainer_cmds = [
            launch_cmd,
            process_node_reports_cmd,
            filter_node_reports_cmd
        ]

        app_log_file = Path(host_log_folder) / "apptainer_out.log"

        # Create the file for logging (truncate if exists, create if not)
        app_log_file.write_text('')

        # Run apptainer commands
        for i, cmd in enumerate(apptainer_cmds):
            app_exec_cmd = (
                f"{exec_cmd}\"{cmd}\" >> {app_log_file} 2>&1"
            )
            print(f"Apptainer command: {app_exec_cmd}")
            out = subprocess.call(app_exec_cmd, shell=True)
            if out != 0:
                print(f"Apptainer command failed with exit code {out} (step {i})")
                return IndividualEvalOut(-float(100+i))

        vpositions_csv_file = host_log_folder / "abe_positions_filtered.csv"
        vehicle_pts = readXyCsv(vpositions_csv_file)

        swimmers_rescued = self.computeSwimmersRescued(vehicle_pts, self.default_swimmer_pts)
        score = swimmers_rescued / len(self.default_swimmer_pts)

        return IndividualEvalOut(score)

    def setupTrialFitnessCsv(self):
        # Setup folder
        folder = self.host_root_folder / ('trial_'+str(self.trial_id))
        folder.mkdir(parents=True, exist_ok=True)
        # Define filepath
        self.fitness_csv_file = folder / 'fitness.csv'

        # Define header columns
        columns = ['generation']
        for i in range(self.population_size):
            columns.append('individual_'+str(i))
            for j in range(self.num_rollouts_per_indivdiual):
                columns.append('individual_'+str(i)+'_rollout_'+str(j))
        # Create empty dataframe
        df = pd.DataFrame(columns=columns)
        # Write only the header to the CSV
        df.to_csv(self.fitness_csv_file, index=False)

    def updateTrialFitnessCsv(self, population):
        # Build out a dictionary of fitnesses
        fit_dict = {
            'gen' : self.gen
        }
        for individual in population:
            # Write the individual's fitness
            ind = 'individual_'+str(individual.temp_id)
            fit_dict[ind] = individual.fitness
            # Write fitness from each rollout
            for r, rfit in enumerate(individual.rollout_fitnesses):
                ind_roll = ind+'_rollout_'+str(r)
                fit_dict[ind_roll] = rfit

        # Convert fit_dict to DataFrame
        df = pd.DataFrame([fit_dict])

        # Append the row to the CSV file
        df.to_csv(self.fitness_csv_file, mode='a', header=False, index=False)

    def evaluateIndividuals(self, individual_eval_ins):
        if self.use_multiprocessing:
            jobs = self.map(self.evaluateIndividual, individual_eval_ins)
            individual_eval_outs = jobs.get()
        else:
            individual_eval_outs = list(self.map(self.evaluateIndividual, individual_eval_ins))
        return individual_eval_outs

    def evaluatePopulation(self, population):
        # Give each individual a temporary id for evaluation
        for i in range(len(population)):
            population[i].temp_id = i

        # Set up directories for evaluating the population at this generation
        self.trial_folder = 'trial_'+str(self.trial_id)
        self.gen_folder = 'gen_'+str(self.gen)

        self.host_gen_folder = self.host_root_folder / self.trial_folder / self.gen_folder
        self.host_gen_folder.mkdir(parents=True, exist_ok=True)

        # Set up the info we need for rollouts
        individual_eval_ins = []
        rollout_seeds = [self.getSeed() for _ in range(self.rpi)]
        for individual in population:
            for rollout_id, seed in enumerate(rollout_seeds):
                individual_eval_ins.append(
                    IndividualEvalIn(individual, seed, rollout_id)
                )

        # Let's evaluate individuals in our rollouts
        individual_eval_outs = self.evaluateIndividuals(individual_eval_ins)

        # Now wrap everything up nicely into summaries
        individual_summaries = self.buildIndividualSummaries(individual_eval_ins, individual_eval_outs)

        return individual_summaries

    def runTrial(self):
        # Check if we are starting with a random seed
        # Handle seed logic
        if self.config_seed is not None:
            self.resetSeed()
            if self.increment_seed_every_trial:
                self.random_seed_val += self.trial_id
            random.seed(self.getSeed())

        # We need a top level fitness file fitness.csv for the trial
        self.setupTrialFitnessCsv()

        # Initialize the population
        population = self.population()

        # Evaluate each individual
        individual_summaries = self.evaluatePopulation(population)

        # Wrap summary information back into each individual
        self.wrapIndividuals(population, individual_summaries)

        # Update fitness.csv
        self.updateTrialFitnessCsv(population)

        # Iterate generations
        for _ in tqdm(range(self.num_generations)):
            # Update gen counter
            self.gen += 1

            # Set the seed for this trial and generation
            if self.config_seed is not None:
                self.resetSeed()
                if self.increment_seed_every_trial:
                    self.random_seed_val += self.trial_id
                self.random_seed_val += self.gen
                random.seed(self.getSeed())

            # Perform selection and mutation
            offspring = self.selectAndMutate(population, individual_summaries)

            # Now population the individuals with the offspring
            population[:] = offspring

            # Evaluate the new population
            individual_summaries[:] = self.evaluatePopulation(population)

            # Wrap summary information back in
            self.wrapIndividuals(population, individual_summaries)

            # Update fitness.csv
            self.updateTrialFitnessCsv(population)

    def run(self):
        for trial_id in range(self.num_trials):
            self.trial_id = trial_id
            self.runTrial()

        if self.use_multiprocessing:
            self.pool.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evolutionary algorithm.")
    parser.add_argument('config_dir', type=str, help='Path to config directory')
    args = parser.parse_args()

    ea = EvolutionaryAlgorithm(args.config_dir)
    ea.run()
