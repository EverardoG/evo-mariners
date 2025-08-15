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
import pickle
import logging
import fcntl
import time

from tqdm import tqdm
import pandas as pd
from shapely.geometry import Point, Polygon
import yaml

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

def parsePolygonString(poly_str):
    """
    Parse a polygon string like 'pts={60,10:-75.5402,-54.2561:-36.9866,-135.58:98.5536,-71.3241}'
    and return a Shapely Polygon object.

    Args:
        poly_str: String representation of polygon points

    Returns:
        shapely.geometry.Polygon: The parsed polygon
    """
    # Extract the points part from the string
    points_part = poly_str.split('pts={')[1].rstrip('}')

    # Split by ':' to get individual points
    point_strings = points_part.split(':')

    # Parse each point (format: "x,y")
    points = []
    for point_str in point_strings:
        x, y = map(float, point_str.split(','))
        points.append((x, y))

    return Polygon(points)

def generateRandomPointsInPolygon(polygon, num_points, seed=None):
    """
    Generate random points within a given polygon using rejection sampling.

    Args:
        polygon: shapely.geometry.Polygon object
        num_points: Number of random points to generate
        seed: Random seed for reproducibility

    Returns:
        List[shapely.geometry.Point]: List of random points within the polygon
    """
    if seed is not None:
        random.seed(seed)

    points = []
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)

    while len(points) < num_points:
        # Generate random point within bounding box
        x = random.uniform(bounds[0], bounds[2])
        y = random.uniform(bounds[1], bounds[3])
        point = Point(x, y)

        # Check if point is within polygon
        if polygon.contains(point):
            points.append(point)

    return points

def generateRandomPointsInCircle(center_x, center_y, radius, num_points, seed=None):
    """
    Generate random points within a circle using polar coordinates.

    Args:
        center_x: X coordinate of circle center
        center_y: Y coordinate of circle center
        radius: Maximum radius from center
        num_points: Number of random points to generate
        seed: Random seed for reproducibility

    Returns:
        List[shapely.geometry.Point]: List of random points within the circle
    """
    if seed is not None:
        random.seed(seed)

    points = []
    for _ in range(num_points):
        # Use sqrt to ensure uniform distribution in area
        r = radius * math.sqrt(random.random())
        theta = random.uniform(0, 2 * math.pi)

        x = center_x + r * math.cos(theta)
        y = center_y + r * math.sin(theta)
        points.append(Point(x, y))

    return points

class ThreadSafeFileHandler(logging.FileHandler):
    """
    A file handler that uses file locking to ensure thread-safe writing
    across multiple processes.
    """
    def emit(self, record):
        """
        Emit a record with file locking to ensure thread safety.
        """
        try:
            if self.stream is None:
                self.stream = self._open()

            # Format the record
            msg = self.format(record)

            # Lock the file for writing
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)

            try:
                self.stream.write(msg + self.terminator)
                self.stream.flush()
            finally:
                # Always release the lock
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)

        except Exception:
            self.handleError(record)

class EvolutionaryAlgorithm():
    def __init__(self, config_filepath: str):
        self.config_filepath = Path(config_filepath)
        self.load_config()

        self.gen = None
        self.trial_id = None
        self.random_seed_val = None

        self.neural_network_size = getSizeOfNet(self.neural_network_structure)
        self.rpi = self.num_rollouts_per_individual

        self.host_root_folder = self.config_filepath.parent
        self.host_root_folder.mkdir(parents=True, exist_ok=True)
        self.trial_folder = None
        self.app_home = Path('/home/moos')
        self.app_root_folder = self.app_home / 'hpc-share'
        self.fitness_csv_file = None

        # Initialize logger
        self.logger = logging.getLogger('EvolutionaryAlgorithm')
        self.logger.setLevel(logging.INFO)

        self.setupMapping()

    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_filepath, 'r') as f:
            config = yaml.safe_load(f)

        # Checkpoint settings
        self.load_checkpoint = config.get('load_checkpoint', True)
        self.delete_previous_checkpoint = config.get('delete_previous_checkpoint', False)

        # Multiprocessing settings
        self.use_multiprocessing = config.get('use_multiprocessing', True)
        self.num_processes = config.get('num_processes', 5)

        # Evolution settings
        self.num_trials = config.get('num_trials', 1)
        self.num_generations = config.get('num_generations', 100)
        self.config_seed = config.get('config_seed', 0)
        self.increment_seed_every_trial = config.get('increment_seed_every_trial', True)

        # Rescue and neural network settings
        self.rescue_observation_radius = config.get('rescue_observation_radius', 100)
        self.neural_network_structure = config.get('neural_network_structure', [8, 10, 5, 2])
        self.neural_network_action_bounds = config.get('neural_network_action_bounds', [[0.0, 1.0], [-180.0, 180.0]])

        # Population settings
        self.population_size = config.get('population_size', 50)
        self.num_rollouts_per_individual = config.get('num_rollouts_per_individual', 2)

        # Selection settings
        self.n_elites = config.get('n_elites', 5)
        self.tournament_size = config.get('tournament_size', 3)

        # Mutation settings
        self.mut_indpb = config.get('mut_indpb', 0.2)
        self.mut_std = config.get('mut_std', 1.0)

        # Parse swimmer spawning configuration
        self.swimmer_spawner_type = config.get('swimmer_spawner', 'fixed')

        if self.swimmer_spawner_type == 'fixed':
            swimmer_config = config.get('swimmer_spawner.fixed.pts', [{'x': 12.0, 'y': -60.0}])
            self.default_swimmer_pts = [Point(pt['x'], pt['y']) for pt in swimmer_config]
        elif self.swimmer_spawner_type == 'rotate':
            swimmer_config = config.get('swimmer_spawner.rotate.pts', [
                [{'x': 12.0, 'y': -60.0}],
                [{'x': 22.0, 'y': -50.0}]
            ])
            self.rotate_swimmer_pts = []
            for rollout in swimmer_config:
                rollout_points = [Point(pt['x'], pt['y']) for pt in rollout]
                self.rotate_swimmer_pts.append(rollout_points)
        elif self.swimmer_spawner_type == 'random':
            self.num_random_swimmers = config.get('swimmer_spawner.random.num', 1)
            self.random_distribution = config.get('swimmer_spawner.random.distribution', 'uniform')

            if self.random_distribution == 'uniform':
                polygon_str = config.get('swimmer_spawner.random.polygon', 'pts={60,10:-75.5402,-54.2561:-36.9866,-135.58:98.5536,-71.3241}')
                self.swimmer_polygon = parsePolygonString(polygon_str)
            elif self.random_distribution == 'circle':
                self.circle_radius = config.get('swimmer_spawner.random.circle.radius', 30)
                circle_center = config.get('swimmer_spawner.random.circle.center', {'x': 0, 'y': 0})
                self.circle_center_x = circle_center['x']
                self.circle_center_y = circle_center['y']

        # Remove old default_swimmer_pts loading if swimmer_spawner is defined
        if 'swimmer_spawner' not in config:
            # Fallback to old configuration for backward compatibility
            swimmer_config = config.get('default_swimmer_pts', [[{'x': 12.0, 'y': -60.0}], [{'x': 22.0, 'y': -50.0}]])
            self.default_swimmer_pts = []
            for rollout in swimmer_config:
                rollout_points = [Point(pt['x'], pt['y']) for pt in rollout]
                self.default_swimmer_pts.append(rollout_points)

        # Population settings
        self.population_size = config.get('population_size', 50)
        self.num_rollouts_per_individual = config.get('num_rollouts_per_individual', 2)

        # Selection settings
        self.n_elites = config.get('n_elites', 5)
        self.tournament_size = config.get('tournament_size', 3)

        # Mutation settings
        self.mut_indpb = config.get('mut_indpb', 0.2)
        self.mut_std = config.get('mut_std', 1.0)

        # Parse vehicle poses
        pose_config = config.get('default_vehicle_poses',
                                [[{'x': 13.0, 'y': -20.0, 'heading': 181.0}],
                                 [{'x': 13.0, 'y': -20.0, 'heading': 181.0}]])
        self.default_vehicle_poses = []
        for rollout in pose_config:
            rollout_poses = [Pose(pose['x'], pose['y'], pose['heading']) for pose in rollout]
            self.default_vehicle_poses.append(rollout_poses)

        # MOOS settings
        self.moos_timewarp = config.get('moos_timewarp', 10)
        self.max_db_uptime = config.get('max_db_uptime', 120)
        self.trim_logs = config.get('trim_logs', True)

        # Apptainer settings
        self.launch_timeout = config.get('launch_timeout', 'auto')
        if self.launch_timeout == 'auto':
            self.launch_timeout = self.max_db_uptime/self.moos_timewarp+60

        self.process_nodes_timeout = config.get('process_nodes_timeout', 'auto')
        if self.process_nodes_timeout == 'auto':
            self.process_nodes_timeout = self.max_db_uptime*2

        self.filter_csv_timeout = config.get('filter_csv_timeout', 'auto')
        if self.filter_csv_timeout == 'auto':
            self.filter_csv_timeout = self.max_db_uptime*2

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

    def saveCheckpoint(self, population, individual_summaries):
        checkpoint_dir = self.trial_folder/('checkpoint_'+str(self.gen)+'.pkl')
        with open(checkpoint_dir, 'wb') as f:
            pickle.dump((population, individual_summaries), f)
        if self.delete_previous_checkpoint:
            checkpoint_dirs = [dir for dir in os.listdir(self.trial_folder) if "checkpoint_" in dir]
            if len(checkpoint_dirs) > 1:
                lower_gen = min( [int(dir.split("_")[-1].split('.')[0]) for dir in checkpoint_dirs] )
                prev_checkpoint_dir = self.trial_folder/('checkpoint_'+str(lower_gen)+'.pkl')
                os.remove(prev_checkpoint_dir)

    def getCheckpointDirs(self):
        return [self.trial_folder/dir for dir in os.listdir(self.trial_folder) if "checkpoint_" in dir]

    def loadCheckpoint(self, checkpoint_dirs):
        checkpoint_dirs.sort(key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
        with open(checkpoint_dirs[-1], 'rb') as f:
            population, individual_summaries = pickle.load(f)
        gen = int(str(checkpoint_dirs[-1]).split('_')[-1].split('.')[0])
        return population, individual_summaries, gen

    def writeNeuralNetworkCsv(self, individual, filepath):
        return writeNeuralNetworkCsv(
            individual.weights,
            self.neural_network_structure,
            self.neural_network_action_bounds,
            filepath
        )

    def getSwimmerPtsForRollout(self, rollout_id, seed=None):
        """
        Get swimmer points for a specific rollout based on the spawning type.

        Args:
            rollout_id: The rollout identifier
            seed: Random seed for random spawning

        Returns:
            List[shapely.geometry.Point]: List of swimmer points for this rollout
        """
        if self.swimmer_spawner_type == 'fixed':
            return self.default_swimmer_pts
        elif self.swimmer_spawner_type == 'rotate':
            # Rotate through the available configurations
            config_index = rollout_id % len(self.rotate_swimmer_pts)
            return self.rotate_swimmer_pts[config_index]
        elif self.swimmer_spawner_type == 'random':
            if self.random_distribution == 'uniform':
                # Generate random points using the polygon
                return generateRandomPointsInPolygon(self.swimmer_polygon, self.num_random_swimmers, seed)
            elif self.random_distribution == 'circle':
                # Generate random points using the circle
                return generateRandomPointsInCircle(
                    self.circle_center_x,
                    self.circle_center_y,
                    self.circle_radius,
                    self.num_random_swimmers,
                    seed
                )
        else:
            # Fallback to old behavior
            if hasattr(self, 'default_swimmer_pts') and len(self.default_swimmer_pts) > rollout_id:
                return self.default_swimmer_pts[rollout_id]
            else:
                return [Point(12.0, -60.0)]  # Default fallback

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
        # Set up logger for this process if it doesn't have handlers
        if not self.logger.handlers:
            log_file = self.trial_folder / 'trial.log'
            file_handler = ThreadSafeFileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Configure logger with process context
        process_id = os.getpid()
        context_prefix = f"PID:{process_id} Gen:{self.gen} Ind:{individual_eval_in.individual.temp_id} Rollout:{individual_eval_in.rollout_id}"

        individual_folder = 'ind_'+str(individual_eval_in.individual.temp_id)
        rollout_folder = 'rollout_'+str(individual_eval_in.rollout_id)

        host_work_folder = self.host_root_folder / self.trial_name / self.gen_folder / individual_folder / rollout_folder
        host_log_folder = host_work_folder / 'logs'
        host_log_folder.mkdir(parents=True, exist_ok=True)

        host_swimmers_txt_file = host_work_folder / 'swimmers.txt'
        host_vpositions_txt_file = host_work_folder / 'vpositions.txt'
        host_neural_net_csv_file = host_work_folder / 'neural_network_abe.csv'

        app_work_folder = self.app_root_folder / self.trial_name / self.gen_folder / individual_folder / rollout_folder
        app_log_folder = app_work_folder / 'logs'
        apptainer_sif_file = getSourceDir() / "apptainer" / "ubuntu_20.04_ivp_2680_learn.sif";

        app_swimmers_txt_file = app_work_folder / 'swimmers.txt'
        app_vpositions_txt_file = app_work_folder / 'vpositions.txt'
        app_neural_net_csv_file = app_work_folder / 'neural_network_abe.csv'

        # Get swimmer points for this rollout
        swimmer_pts = self.getSwimmerPtsForRollout(individual_eval_in.rollout_id, individual_eval_in.seed)

        writeSwimmersTxt(swimmer_pts, host_swimmers_txt_file)
        # Use default vehicle poses for backward compatibility or add similar logic for vehicle poses if needed
        vehicle_poses = self.default_vehicle_poses[individual_eval_in.rollout_id] if hasattr(self, 'default_vehicle_poses') and len(self.default_vehicle_poses) > individual_eval_in.rollout_id else [Pose(13.0, -20.0, 181.0)]
        writeVpositionsTxt(vehicle_poses, host_vpositions_txt_file)
        self.writeNeuralNetworkCsv(individual_eval_in.individual, host_neural_net_csv_file)

        # Build launch arguments
        launch_args = [
            str(self.moos_timewarp),  # timewarp
            "--xlaunched",
            f"--logdir={app_log_folder}/",
            f"--neural_network_dir={app_work_folder}/",
            "--uMayFinish",
            f"--max_db_uptime={self.max_db_uptime}",
            "--nogui",
            "--rescuebehavior=NeuralNetwork",
            # "--rescuebehavior=FollowCOM",
            "--autodeploy",
            f"--swim_file={app_swimmers_txt_file}",
            f"--vpositions={app_work_folder}/vpositions.txt",
            "--nostamp",
            f"--rescue_observation_radius={self.rescue_observation_radius}"
        ]

        # Conditionally add --trim flag
        if self.trim_logs:
            launch_args.append("--trim")

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
            # "--fakeroot",
            f"--bind {self.host_root_folder}:{self.app_root_folder}",
            "--writable-tmpfs",
            str(apptainer_sif_file),
            "/bin/bash -c "
        ]
        exec_cmd = " ".join(exec_pieces)

        process_node_reports_cmd = (
            f"process_node_reports {app_log_folder}/XLOG_SHORESIDE/XLOG_SHORESIDE.alog {app_log_folder}/"
        )
        # cut_last_lines is necessary in case the node report wasn't complete in the last line
        cut_last_lines_cmd = (
            f"sed -i '$d;$d' {app_log_folder}/abe_positions.csv"
        )
        filter_node_reports_cmd = (
            f"csv_filter_duplicate_rows {app_log_folder}/abe_positions.csv {app_log_folder}/abe_positions_filtered.csv"
        )
        apptainer_cmds = [
            launch_cmd,
            process_node_reports_cmd,
            cut_last_lines_cmd,
            filter_node_reports_cmd
        ]
        timeouts = [
            self.launch_timeout,
            self.process_nodes_timeout,
            100, # cut_last_lines_timeout
            self.filter_csv_timeout
        ]

        app_log_file = Path(host_log_folder) / "apptainer_out.log"

        # Create the file for logging (truncate if exists, create if not)
        app_log_file.write_text('')

        # Run apptainer commands
        for i, (cmd, timeout) in enumerate(zip(apptainer_cmds, timeouts)):
            app_exec_cmd = (
                f"{exec_cmd}\"{cmd}\" >> {app_log_file} 2>&1"
            )

            self.logger.info(f"{context_prefix} - Running apptainer command {i+1}/4: {cmd}")

            try:
                out = subprocess.call(app_exec_cmd, shell=True, timeout=timeout)
                if out == 0:
                    self.logger.info(f"{context_prefix} - Apptainer command {i+1}/4 completed successfully")
                else:
                    self.logger.error(f"{context_prefix} - Apptainer command {i+1}/4 failed with exit code {out}")
                    return IndividualEvalOut(-float(100+i))
            except subprocess.TimeoutExpired:
                self.logger.error(f"{context_prefix} - Apptainer command {i+1}/4 timed out after {timeout} seconds")
                return IndividualEvalOut(-float(100+i))

        vpositions_csv_file = host_log_folder / "abe_positions_filtered.csv"
        vehicle_pts = readXyCsv(vpositions_csv_file)

        swimmers_rescued = self.computeSwimmersRescued(vehicle_pts, swimmer_pts)
        score = swimmers_rescued / len(swimmer_pts)

        return IndividualEvalOut(score)

    def setupTrialLogger(self):
        """Set up trial-specific logging to trial.log file"""
        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create thread-safe file handler for trial.log
        log_file = self.trial_folder / 'trial.log'
        file_handler = ThreadSafeFileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)

        self.logger.info(f"Starting trial {self.trial_id}")

    def setupTrialFitnessCsv(self):
        # Define filepath
        self.fitness_csv_file = self.trial_folder / 'fitness.csv'

        # Define header columns
        columns = ['generation']
        for i in range(self.population_size):
            columns.append('individual_'+str(i))
            for j in range(self.num_rollouts_per_individual):
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
        self.trial_name = 'trial_'+str(self.trial_id)
        self.gen_folder = 'gen_'+str(self.gen)

        self.host_gen_folder = self.host_root_folder / self.trial_name / self.gen_folder
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
        # Setup folder
        self.trial_folder = self.host_root_folder / ('trial_'+str(self.trial_id))
        self.trial_folder.mkdir(parents=True, exist_ok=True)

        # Setup trial-specific logging
        self.setupTrialLogger()

        # Check if we are loading a checkpoint or initializing from scratch
        if self.load_checkpoint and len(checkpoint_dirs := self.getCheckpointDirs()) > 0:
            population, individual_summaries, self.gen = self.loadCheckpoint(checkpoint_dirs)
        else:
            # Reset generation counter at start of each trial
            self.gen = 0

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

            # Save it
            self.saveCheckpoint(population, individual_summaries)

        # Iterate generations
        for _ in tqdm(range(self.num_generations-self.gen)):
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

            # Save it
            self.saveCheckpoint(population, individual_summaries)

    def run(self):
        for trial_id in range(self.num_trials):
            self.trial_id = trial_id
            self.runTrial()

        if self.use_multiprocessing:
            self.pool.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evolutionary algorithm.")
    parser.add_argument('config_filepath', type=str, help='Path to config YAML file')
    args = parser.parse_args()

    ea = EvolutionaryAlgorithm(args.config_filepath)
    ea.run()
