import argparse
import pandas as pd
import re
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort team fitness scores by rank. Takes fitness CSV files and sorts entries in each row based on score, with higher scores in leftmost columns."
    )
    parser.add_argument(
        "input_file",
        help="Input CSV file path"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV file path (default: same directory as input, filename prefixed with 'sorted_')",
        default=None,
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Generate output filename if not provided
    if args.output is None:
        input_dir = os.path.dirname(args.input_file)
        input_filename = os.path.basename(args.input_file)
        output_filename = f"sorted_{input_filename}"
        args.output = os.path.join(input_dir, output_filename)

    try:
        # Read the CSV file
        df = pd.read_csv(args.input_file)

        # Find team fitness columns and extract team IDs
        team_fitness_pattern = r'team_(\d+)_team_fitness'
        team_fitness_cols = []
        team_ids = []

        for col in df.columns:
            match = re.match(team_fitness_pattern, col)
            if match:
                team_fitness_cols.append(col)
                team_ids.append(int(match.group(1)))

        if not team_fitness_cols:
            print("Error: No team_<digits>_team_fitness columns found in the input file", file=sys.stderr)
            sys.exit(1)

        # Verify generation column exists
        if 'generation' not in df.columns:
            print("Error: 'generation' column not found in the input file", file=sys.stderr)
            sys.exit(1)

        # Check for missing or non-numeric values in fitness columns
        for col in team_fitness_cols:
            if df[col].isna().any():
                print(f"Error: Missing values found in column '{col}'", file=sys.stderr)
                sys.exit(1)
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Error: Non-numeric values found in column '{col}'", file=sys.stderr)
                sys.exit(1)

        # Create output dataframe starting with generation column
        result_df = pd.DataFrame()
        result_df['generation'] = df['generation']

        # Sort teams by fitness for each row
        num_teams = len(team_fitness_cols)

        for i in range(len(df)):
            # Get fitness values for this generation
            fitness_values = df.iloc[i][team_fitness_cols].values

            # Sort indices by fitness (descending order)
            sorted_indices = sorted(range(len(fitness_values)), key=lambda x: fitness_values[x], reverse=True)

            # Assign sorted IDs and values to new columns (ID first, then fitness)
            for rank, orig_index in enumerate(sorted_indices):
                id_col_name = f'sorted_team_{rank}_id'
                fitness_col_name = f'sorted_team_{rank}'

                if id_col_name not in result_df.columns:
                    result_df[id_col_name] = 0
                if fitness_col_name not in result_df.columns:
                    result_df[fitness_col_name] = 0.0

                result_df.at[i, id_col_name] = team_ids[orig_index]
                result_df.at[i, fitness_col_name] = fitness_values[orig_index]

        # Write output file
        result_df.to_csv(args.output, index=False)
        print(f"Sorted fitness data written to: {args.output}")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
