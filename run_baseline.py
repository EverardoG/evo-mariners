import os
import argparse
from pathlib import Path
import csv
from shapely.geometry import Point
import logging
import subprocess
import yaml
import multiprocessing
from tqdm import tqdm
import math

class BaselineEvaluator:
    def __init__(self, trial_path, max_generation):
        self.trial_path = Path(trial_path)
        self.max_generation = max_generation

        # Load configuration from the trial's config file
        config_file = self.trial_path.parent / "config.yaml"
        self.load_config(config_file)

        # Set up paths similar to CCEA
        self.host_root_folder = self.trial_path.parent.parent
        self.app_home = Path('/home/moos')
        self.app_root_folder = self.app_home / 'hpc-share'

        # Set up baseline output directory as SIBLING of trial_0
        trial_name = self.trial_path.name  # e.g., "trial_0"
        baseline_name = f"baseline_{trial_name}"
        self.baseline_folder = self.trial_path.parent / baseline_name  # Changed back to sibling of trial_0
        self.baseline_folder.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = logging.getLogger('BaselineEvaluator')
        self.logger.setLevel(logging.INFO)

    def load_config(self, config_filepath):
        """Load relevant configuration from YAML file"""
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)

        # Load only the configuration we need for baseline evaluation
        self.moos_timewarp = config.get('moos_timewarp', 10)
        self.max_db_uptime = config.get('max_db_uptime', 120)
        self.trim_logs = config.get('trim_logs', True)
        self.rescue_observation_radius = config.get('rescue_observation_radius', 100)
        self.num_swimmer_sectors = config.get('num_swimmer_sectors', 8)

        # Timeout settings
        self.launch_timeout = config.get('launch_timeout', 'auto')
        if self.launch_timeout == 'auto':
            self.launch_timeout = self.max_db_uptime/self.moos_timewarp+60

        self.process_nodes_timeout = config.get('process_nodes_timeout', 'auto')
        if self.process_nodes_timeout == 'auto':
            self.process_nodes_timeout = self.max_db_uptime*2

        self.filter_csv_timeout = config.get('filter_csv_timeout', 'auto')
        if self.filter_csv_timeout == 'auto':
            self.filter_csv_timeout = self.max_db_uptime*2

        # Multiprocessing settings
        self.use_multiprocessing = config.get('use_multiprocessing', True)
        self.num_processes = config.get('num_processes', 5)

        # Parse polygon string for swimmer rescue computation
        self.polygon_str = config.get(
            'swimmer_spawner.polygon',
            'pts={60,10:-75.5402,-54.2561:-36.9866,-135.58:98.5536,-71.3241}'
        )

    def writeSwimmersTxt(self, swimmer_pts, polygon_str, filepath):
        """
        Write swimmers data to a text file.

        Args:
            swimmer_pts: List of shapely Point objects representing swimmer positions
            polygon_str: moos-ivp formatted string for swim region
            filepath: pathlib.Path object for the output file

        Raises:
            IOError: If the file cannot be opened or written to
            Exception: For any other errors during file operations
        """
        with open(filepath, 'w') as f:
            # Write the top line
            f.write(f"poly = {polygon_str}\n")

            # Write swimmer data
            for i, swimmer_pt in enumerate(swimmer_pts):
                # Format: swimmer = name=p01, x=13, y=10
                f.write(f"swimmer = name=p{i+1:02d}, x={swimmer_pt.x}, y={swimmer_pt.y}\n")

    def writeVpositionsTxt(self, vehicle_poses, filepath):
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

    def readXyCsv(self, filepath):
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

    def evaluateBaselineRollout(self, rollout_info):
        """
        Evaluate a baseline rollout using existing swimmers.txt and vpositions.txt files
        """
        generation = rollout_info['generation']
        rollout_id = rollout_info['rollout_id']
        swimmers_file = rollout_info['swimmers_file']
        vpositions_file = rollout_info['vpositions_file']

        process_id = os.getpid()
        context_prefix = f"PID:{process_id} Gen:{generation} Baseline Rollout:{rollout_id}"

        # Set up baseline output directories
        baseline_gen_folder = self.baseline_folder / f'gen_{generation}'
        baseline_rollout_folder = baseline_gen_folder / f'rollout_{rollout_id}'
        baseline_log_folder = baseline_rollout_folder / 'logs'
        baseline_log_folder.mkdir(parents=True, exist_ok=True)

        # Load and rewrite files (to ensure consistency)
        swimmer_pts = self.load_swimmers_from_file(swimmers_file)
        vehicle_poses = self.load_vehicles_from_file(vpositions_file)

        baseline_swimmers_file = baseline_rollout_folder / 'swimmers.txt'
        baseline_vpositions_file = baseline_rollout_folder / 'vpositions.txt'

        self.writeSwimmersTxt(swimmer_pts, self.polygon_str, baseline_swimmers_file)
        self.writeVpositionsTxt(vehicle_poses, baseline_vpositions_file)

        # Set up apptainer paths - FIXED to use correct binding structure
        # The baseline folder is at: parent_directory/baseline_trial_0/
        # When bound to /home/moos/hpc-share, it becomes: /home/moos/hpc-share/baseline_trial_0/
        baseline_relative_path = self.baseline_folder.relative_to(self.host_root_folder)
        app_baseline_folder = self.app_root_folder / baseline_relative_path
        app_rollout_folder = app_baseline_folder / f'gen_{generation}' / f'rollout_{rollout_id}'
        app_log_folder = app_rollout_folder / 'logs'
        app_swimmers_file = app_rollout_folder / 'swimmers.txt'
        app_vpositions_file = app_rollout_folder / 'vpositions.txt'

        apptainer_sif_file = getSourceDir() / "apptainer" / "ubuntu_20.04_ivp_2680_learn.sif"

        # Create rollout logger
        rollout_log_file = baseline_log_folder / 'rollout.log'
        rollout_logger = logging.getLogger(f'BaselineRolloutLogger_{process_id}_{rollout_id}')
        rollout_logger.setLevel(logging.INFO)

        # Clear any existing handlers
        for handler in rollout_logger.handlers[:]:
            handler.close()
            rollout_logger.removeHandler(handler)

        file_handler = logging.FileHandler(rollout_log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - PID:%(process)d - %(message)s')
        file_handler.setFormatter(formatter)
        rollout_logger.addHandler(file_handler)

        # Build launch arguments (NO neural network arguments)
        launch_args = [
            str(self.moos_timewarp),
            "--xlaunched",
            f"--logdir={app_log_folder}/",
            "--uMayFinish",
            f"--max_db_uptime={self.max_db_uptime}",
            "--nogui",
            "--autodeploy",
            f"--swim_file={app_swimmers_file}",
            f"--vpositions={app_vpositions_file}",
            "--nostamp",
            f"--rescue_observation_radius={self.rescue_observation_radius}",
            f"--r{len(vehicle_poses)}",
            f"--r_swimmer_sectors={self.num_swimmer_sectors}"
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
            f"--bind {self.host_root_folder}:{self.app_root_folder}",
            "--writable-tmpfs",
            str(apptainer_sif_file),
            "/bin/bash -c "
        ]
        exec_cmd = " ".join(exec_pieces)

        # Build command sequence (similar to CCEA but without neural network processing)
        process_node_reports_cmd = (
            f"process_node_reports {app_log_folder}/XLOG_SHORESIDE/XLOG_SHORESIDE.alog {app_log_folder}/"
        )
        apptainer_cmds = [
            launch_cmd,
            process_node_reports_cmd
        ]
        timeouts = [
            self.launch_timeout,
            self.process_nodes_timeout
        ]

        # Add CSV processing commands
        for i in range(len(vehicle_poses)):
            cut_last_lines_cmd = (
                f"sed -i '$d;$d' {app_log_folder}/abe{i+1}_positions.csv"
            )
            apptainer_cmds.append(cut_last_lines_cmd)
            timeouts.append(100)

        csv_merge_files_cmd = (
            f"csv_merge_files {app_log_folder}/"
        )
        apptainer_cmds.append(csv_merge_files_cmd)
        timeouts.append(500)

        filter_node_reports_cmd = (
            f"csv_filter_duplicate_rows {app_log_folder}/team_positions.csv {app_log_folder}/team_positions_filtered.csv"
        )
        apptainer_cmds.append(filter_node_reports_cmd)
        timeouts.append(self.filter_csv_timeout)

        app_log_file = baseline_log_folder / "apptainer_out.log"
        with open(app_log_file, 'w') as f:
            f.write(' ')

        # Run apptainer commands
        for i, (cmd, timeout) in enumerate(zip(apptainer_cmds, timeouts)):
            app_exec_cmd = (
                f"{exec_cmd}\"{cmd}\" >> {app_log_file} 2>&1"
            )

            rollout_logger.info(f"{context_prefix} - Running apptainer command {i+1}/{len(apptainer_cmds)}: {cmd}")

            try:
                out = subprocess.call(app_exec_cmd, shell=True, timeout=timeout)
                if out == 0:
                    rollout_logger.info(f"{context_prefix} - Apptainer command {i+1}/{len(apptainer_cmds)} completed successfully")
                else:
                    rollout_logger.error(f"{context_prefix} - Apptainer command {i+1}/{len(apptainer_cmds)} failed with exit code {out}")
                    team_fitness = -float(100+i)
                    for handler in rollout_logger.handlers[:]:
                        handler.close()
                        rollout_logger.removeHandler(handler)
                    return {
                        'generation': generation,
                        'rollout_id': rollout_id,
                        'team_fitness': team_fitness
                    }
            except subprocess.TimeoutExpired:
                rollout_logger.error(f"{context_prefix} - Apptainer command {i+1}/{len(apptainer_cmds)} timed out after {timeout} seconds")
                team_fitness = -float(100+i)
                for handler in rollout_logger.handlers[:]:
                    handler.close()
                    rollout_logger.removeHandler(handler)
                return {
                    'generation': generation,
                    'rollout_id': rollout_id,
                    'team_fitness': team_fitness
                }

        # Read results and compute fitness
        vpositions_csv_file = baseline_log_folder / "team_positions_filtered.csv"
        vehicle_pts = self.readXyCsv(vpositions_csv_file)

        swimmers_rescued = self.computeSwimmersRescued(vehicle_pts, swimmer_pts)
        team_score = swimmers_rescued / len(swimmer_pts)

        # Clean up logger
        for handler in rollout_logger.handlers[:]:
            handler.close()
            rollout_logger.removeHandler(handler)

        return {
            'generation': generation,
            'rollout_id': rollout_id,
            'team_fitness': team_score
        }

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

    def crawl_trial_rollouts(self):
        """
        Crawl the CCEA trial directory to extract all rollout information.
        Returns a dictionary with generation -> list of rollout info
        """
        rollouts_by_gen = {}

        # Iterate through generations (gen_0, gen_1, etc.)
        for gen_num in range(self.max_generation + 1):
            gen_folder = f"gen_{gen_num}"
            gen_path = self.trial_path / gen_folder

            if not gen_path.exists():
                self.logger.warning(f"Generation folder {gen_folder} not found, stopping at generation {gen_num-1}")
                break

            rollouts_by_gen[gen_num] = []

            # Look for team_0 folders (ignoring the individual indices part)
            team_folders = [f for f in gen_path.iterdir() if f.is_dir() and f.name.startswith('team_0')]

            if not team_folders:
                self.logger.warning(f"No team_0 folders found in {gen_folder}")
                continue

            # Use the first team_0 folder we find
            team_folder = team_folders[0]

            # Find all rollout folders in this team
            rollout_folders = [f for f in team_folder.iterdir() if f.is_dir() and f.name.startswith('rollout_')]
            rollout_folders.sort(key=lambda x: int(x.name.split('_')[1]))

            for rollout_folder in rollout_folders:
                rollout_id = int(rollout_folder.name.split('_')[1])
                swimmers_file = rollout_folder / 'swimmers.txt'
                vpositions_file = rollout_folder / 'vpositions.txt'

                if swimmers_file.exists() and vpositions_file.exists():
                    rollout_info = {
                        'generation': gen_num,
                        'rollout_id': rollout_id,
                        'swimmers_file': swimmers_file,
                        'vpositions_file': vpositions_file,
                        'original_rollout_folder': rollout_folder
                    }
                    rollouts_by_gen[gen_num].append(rollout_info)
                else:
                    self.logger.warning(f"Missing swimmers.txt or vpositions.txt in {rollout_folder}")

        return rollouts_by_gen

    def load_swimmers_from_file(self, swimmers_file):
        """Load swimmer points from a swimmers.txt file"""
        swimmer_pts = []
        with open(swimmers_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('swimmer = '):
                    # Parse: swimmer = name=p01, x=13, y=10
                    parts = line.split(', ')
                    x_val = None
                    y_val = None
                    for part in parts:
                        if part.startswith('x='):
                            x_val = float(part.split('=')[1])
                        elif part.startswith('y='):
                            y_val = float(part.split('=')[1])
                    if x_val is not None and y_val is not None:
                        swimmer_pts.append(Point(x_val, y_val))
        return swimmer_pts

    def load_vehicles_from_file(self, vpositions_file):
        """Load vehicle poses from a vpositions.txt file"""
        vehicle_poses = []
        with open(vpositions_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse: x=13.0,y=-20.0,heading=181.0
                    x_val = None
                    y_val = None
                    heading_val = None
                    parts = line.split(',')
                    for part in parts:
                        if part.startswith('x='):
                            x_val = float(part.split('=')[1])
                        elif part.startswith('y='):
                            y_val = float(part.split('=')[1])
                        elif part.startswith('heading='):
                            heading_val = float(part.split('=')[1])
                    if all(val is not None for val in [x_val, y_val, heading_val]):
                        vehicle_poses.append(Pose(x_val, y_val, heading_val))
        return vehicle_poses

    def run_baseline_evaluation(self):
        """Main method to run baseline evaluation across all generations"""
        # Set up console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info("Starting baseline evaluation")
        self.logger.info(f"Evaluating generations 0 to {self.max_generation}")

        # Crawl trial rollouts
        rollouts_by_gen = self.crawl_trial_rollouts()

        if not rollouts_by_gen:
            self.logger.error("No rollouts found in trial directory")
            return

        # Set up fitness CSV file
        fitness_csv_file = self.baseline_folder / 'fitness.csv'

        # Initialize CSV with headers
        max_rollouts = max(len(rollouts) for rollouts in rollouts_by_gen.values()) if rollouts_by_gen else 0
        headers = ['generation'] + [f'rollout_{i}_team_fitness' for i in range(max_rollouts)]

        with open(fitness_csv_file, 'w') as f:
            f.write(','.join(headers) + '\n')

        # Set up progress bar for all generations
        all_rollouts = []
        for gen_num in sorted(rollouts_by_gen.keys()):
            all_rollouts.extend(rollouts_by_gen[gen_num])

        total_rollouts = len(all_rollouts)
        self.logger.info(f"Found {total_rollouts} total rollouts across {len(rollouts_by_gen)} generations")

        # Process rollouts sequentially for now (multiprocessing has issues with instance methods)
        self.logger.info("Using sequential processing")
        rollout_results = []
        with tqdm(all_rollouts, desc="Evaluating baseline rollouts") as pbar:
            for rollout_info in pbar:
                result = self.evaluateBaselineRollout(rollout_info)
                rollout_results.append(result)

        # Organize results by generation
        results_by_gen = {}
        for result in rollout_results:
            gen = result['generation']
            if gen not in results_by_gen:
                results_by_gen[gen] = []
            results_by_gen[gen].append(result)

        # Write results to CSV
        with open(fitness_csv_file, 'a') as f:
            for gen_num in sorted(results_by_gen.keys()):
                gen_results = sorted(results_by_gen[gen_num], key=lambda x: x['rollout_id'])
                row = [str(gen_num)]
                for i in range(max_rollouts):
                    if i < len(gen_results):
                        row.append(str(gen_results[i]['team_fitness']))
                    else:
                        row.append('')
                f.write(','.join(row) + '\n')

        self.logger.info(f"Baseline evaluation completed. Results saved to {fitness_csv_file}")
        self.print_summary_statistics(results_by_gen)

    def print_summary_statistics(self, results_by_gen):
        """Print summary statistics of baseline evaluation"""
        self.logger.info("=== Baseline Evaluation Summary ===")

        for gen_num in sorted(results_by_gen.keys()):
            gen_results = results_by_gen[gen_num]
            fitnesses = [r['team_fitness'] for r in gen_results if r['team_fitness'] >= 0]

            if fitnesses:
                avg_fitness = sum(fitnesses) / len(fitnesses)
                min_fitness = min(fitnesses)
                max_fitness = max(fitnesses)

                self.logger.info(f"Generation {gen_num}: {len(fitnesses)} rollouts, "
                               f"Avg: {avg_fitness:.3f}, Min: {min_fitness:.3f}, Max: {max_fitness:.3f}")
            else:
                self.logger.info(f"Generation {gen_num}: No successful rollouts")

# Add missing utility functions and classes
def getSourceDir():
    """Returns the absolute directory of this python file"""
    return Path(__file__).parent.resolve()

class Pose:
    """Simple pose class for vehicle positions"""
    def __init__(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Run baseline evaluation on CCEA trial data")
    parser.add_argument('trial_path', type=str, help='Path to CCEA trial directory')
    parser.add_argument('--max_generation', type=int, required=True,
                       help='Maximum generation to evaluate (inclusive)')

    args = parser.parse_args()

    # Create and run baseline evaluator
    evaluator = BaselineEvaluator(args.trial_path, args.max_generation)
    evaluator.run_baseline_evaluation()

if __name__ == '__main__':
    main()
