import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="convert node attributes")
    parser.add_argument('--input_path', default='', type=str, help='Input graph file path')
    parser.add_argument('--num_convert', default=100, type=int, help='Node attr to convert')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    tmp = np.load(os.path.join(args.input_path, "{}.npy".format(0)))
    num_nodes, num_features = tmp.shape
    all_mtrx = np.zeros((239,num_features*args.num_convert))
    count = 0
    for i in range(args.num_convert):
        file_name = os.path.join(args.input_path, "{}.npy".format(i))
        x = np.load(file_name)
        
        all_mtrx[:,num_features*i:num_features*i+num_features]=x 
        count += 1  

    save_path = os.path.join(args.input_path, "consolidated_attr.npy")
    print("=> Save Path: ",save_path)
    np.save(save_path,all_mtrx)

if __name__ == "__main__":
    main() 