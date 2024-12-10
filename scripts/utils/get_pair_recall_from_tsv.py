def get_recall(int_list, query_id, querytype):
    """
    Calculate recall for a query based on its query ID and type.
    """
    if querytype == 't2i':
        # Recall is 1 if query_id // 5 (as integer) is in the result set
        recall = 1 if int(query_id // 5) in int_list else 0
        # print(f"[DEBUG] t2i Recall for query_id {query_id}: {recall}")
        return recall
    elif querytype == 'i2t':
        # Recall is 1 if any of n*5, n*5+1, ..., n*5+4 is in the result set
        start = query_id * 5
        recall = 1 if any(x in int_list for x in range(start, start + 5)) else 0
        # print(f"[DEBUG] i2t Recall for query_id {query_id}: {recall}")
        return recall
    else:
        raise ValueError(f"Unknown query type: {querytype}")

def find_qps_line(original_file, querytype, beam_width):
    """
    Find the QPS line from the original file based on the query type and beam width.
    If query type is 't2i', return the first matching line.
    If query type is 'i2t', return the second matching line.
    """
    print(f"[DEBUG] Finding QPS line in {original_file} for beam_width {beam_width} and querytype {querytype}")
    with open(original_file, 'r') as orig_file:
        matching_lines = []
        for line in orig_file:
            parts = line.split()
            if int(parts[0]) == beam_width:
                matching_lines.append(line.strip())

        if len(matching_lines) != 2:
            print(f"[ERROR] Expected 2 lines with beam width {beam_width}, found {len(matching_lines)}")
            raise ValueError(f"Expected 2 lines with beam width {beam_width}, found {len(matching_lines)}")

        # Return the appropriate line based on query type
        selected_line = matching_lines[0] if querytype == 't2i' else matching_lines[1]
        print(f"[DEBUG] Selected QPS line: {selected_line}")
        return selected_line

def process_file(input_file, output_file, skip_lines, querytype, original_file):
    accum_recall = 0
    query_id = 0

    print(f"[DEBUG] Processing input file: {input_file}")
    try:
        with open(input_file, 'r') as infile:
            while True:
                # Increment query_id

                # Skip predefined number of lines
                for _ in range(skip_lines):
                    skipped_line = infile.readline()
                    if skipped_line == '':
                        # End of file reached during skipping
                        print(f"[DEBUG] End of file reached during skipping for query_id {query_id}.")
                        break

                # Read the next line of integers
                line = infile.readline()
                if not line.strip():
                    # End of file or empty line
                    print(f"[DEBUG] End of file or empty line encountered for query_id {query_id}.")
                    break

                # Convert the line into a list of integers
                try:
                    int_list = list(map(int, line.split()))
                except ValueError:
                    print(f"[ERROR] Error parsing line: {line.strip()}")
                    continue

                # Calculate recall using the provided function
                recall = get_recall(int_list, query_id, querytype)

                # Accumulate the recall
                accum_recall += recall
                query_id += 1
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_file}")
        return

    # Calculate average recall
    avg_recall = accum_recall / query_id if query_id > 0 else 0
    print(f"[DEBUG] Average Recall for {input_file}: {avg_recall:.6f}")

    # Append the updated line to the output file
    try:
        beam_width = int(input_file.split('_')[-1].split('.')[0])  # Extract beam width from filename
        qps_line = find_qps_line(original_file, querytype, beam_width)
        parts = qps_line.split()
        updated_line = f"{parts[0]}\t{avg_recall:.6f}\t{parts[2]}\t{parts[3]}\n"

        print(f"[DEBUG] Writing updated line to output file: {updated_line.strip()}")
        with open(output_file, 'a') as outfile:
            outfile.write(updated_line)
    except Exception as e:
        print(f"[ERROR] Failed to write to output file: {e}")

# Example usage
# for i in range(1, 6):
#     for result_type in ['result_no_adaptive_expansion', 'result_enable_adaptive_expansion', 'result_enable_reverse_adaptive_expansion']:
#         for querytype in ['t2i', 'i2t']:
#             original_file = f"/mnt/dive/{i}/{result_type}/aggregated_results.txt"  # Path to the original file
#             for bw in [20, 40, 80, 120, 160, 200, 400, 600, 800, 1000]:
#                 input_file = f"/mnt/dive/{i}/{result_type}/{querytype}_Roar_35_{bw}.tsv"
#                 output_file = f"/mnt/dive/{i}/{result_type}/aggregated_result_origin.txt"
#                 process_file(input_file, output_file, i + 2, querytype, original_file)
for i in range(2, 6):
    for result_type in ['result_temp']:
        for querytype in ['t2i', 'i2t']:
            original_file = f"/mnt/dive/{i}/{result_type}/aggregated_results.txt"  # Path to the original file
            for bw in [20, 40, 80, 120, 160, 200]:
                input_file = f"/mnt/dive/{i}/{result_type}/{querytype}_Roar_35_{bw}.tsv"
                output_file = f"/mnt/dive/{i}/{result_type}/aggregated_result_origin.txt"
                process_file(input_file, output_file, i + 2, querytype, original_file)
