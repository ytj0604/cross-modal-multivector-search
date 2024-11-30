def process_file(file_path, skip_lines=1):
    accum = 0.0  # Initialize the accumulator for latencies
    with open(file_path, 'r') as f:
        lines = f.readlines()  # Read all lines

        i = 0
        while i < len(lines):
            # Process the line for latency
            line = lines[i].strip()
            parts = line.split()

            if len(parts) >= 2:
                # Parse the integer and float
                try:
                    latency = float(parts[1])  # Second value is the latency
                    accum += latency
                except ValueError:
                    print(f"Error parsing latency from line: {line}")

            # Skip the specified number of lines
            i += (1 + skip_lines)

    print(f"{accum}")


# Example usage
# Save the input text to a file (e.g., "data.txt") and replace 'data.txt' with your file path
#iterate over 20, 40, 80, 120, 160, 200, 400, 600, 800, 1000
for n in [2, 3, 4, 5]:
    for st in ["t2i", "i2t"]:
        for i in [20, 40, 80, 120, 160, 200, 400, 600, 800, 1000]:
            file_path = f"/mnt/dive/{n}/result_shared_visited_list/{st}_Roar_35_{i}.tsv"
            process_file(file_path, skip_lines=n)