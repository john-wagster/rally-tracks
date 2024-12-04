#!/usr/bin/env python3

#```bash
#python3 _tools/parse_fvec_to_json.py data/corpus-dbpedia-entity-E5-small-0.fvec data/corpus-dbpedia-entity-E5-small-1.fvec ... data/quora_e5_small/corpus-dbpedia-entity-E5-small.json
#````

import json
import sys
import struct

dims = 384
max = 150  # None for all

def read_fvecs(files, max=None):
    for f in files:
        print(f"Reading from {f}.")
        with open(f, 'rb') as handle:
            check_dims = struct.unpack('<i', handle.read(4))[0]
            print(f"dims check: {check_dims}")
            exit_while = True
            count = 0
            while(exit_while and count < max):
                count+=1
                vector = []
                for i in range(0, dims):
                    bytes = handle.read(4)
                    if len(bytes) < 4:
                        exit_while = False  # break out of while loop to continue to next file
                        break  # out of for loop
                    vector.append(struct.unpack('<f', bytes)[0])
                yield vector

def write_json(file, vectors):
    with open(file, 'w') as handle:
        for vector in vectors:
            handle.write(json.dumps({"vector": vector}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_files = sys.argv[1:-1]
    output_file = sys.argv[-1]
    print(f"Starting conversion to rally json for: {input_files} to {output_file}")
    vectors_promise = read_fvecs(input_files, max)
    write_json(output_file, vectors_promise)
    print("Conversion complete")
