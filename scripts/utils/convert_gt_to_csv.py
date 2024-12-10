import struct
import csv

def read_groundtruth_file(gt_file, csv_file):
    with open(gt_file, "rb") as f:
        # Read the header
        npts = struct.unpack('i', f.read(4))[0]
        k = struct.unpack('i', f.read(4))[0]
        
        # Read the nearest neighbor IDs
        neighbor_ids = []
        for _ in range(npts * k):
            neighbor_ids.append(struct.unpack('I', f.read(4))[0])
        
        # Read the distances
        distances = []
        for _ in range(npts * k):
            distances.append(struct.unpack('f', f.read(4))[0])
    
    # Write to CSV
    with open(csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(["QueryID", "NeighborID", "Distance"])
        
        # Write the ground truth
        for i in range(npts):
            for j in range(k):
                query_id = i
                neighbor_id = neighbor_ids[i * k + j]
                distance = distances[i * k + j]
                csv_writer.writerow([query_id, neighbor_id, distance])

# Example usage
gt_file = "/mnt/dive/i2i.gt.bin"  # Replace with your ground truth file path
csv_file = "/mnt/tjyoon_roargraph_expr/i2i.gt.csv"  # Replace with your desired output CSV file path
read_groundtruth_file(gt_file, csv_file)
