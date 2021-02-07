import os

import torch
import networkx as nx
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import re
import glob
import tqdm 
import pickle
import argparse
import CTDNE
import numpy as np
import errno
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l 

def arg_parser():
    parser = argparse.ArgumentParser(description='CTDNE Configuration')
    parser.add_argument("--input_path_prefix",dest='input_path_prefix',type=str,help="Prefix for Saved Pkl Graphs",default="/z/pujat/nn_homology/data/")
    parser.add_argument("--embedding_save_path",dest='embedding_save_path',type=str,help='Where to save the stored embeddings',default='/z/pujat/nn_homology/data/embeddings')
    parser.add_argument("--dimensions",dest='dimensions',type=int,help='Dimension of embedding',default=64)
    parser.add_argument("--workers",dest='workers',type=int,help='Num workers', default=4)
    parser.add_argument("--time_steps",dest='time_steps',type=int,help='Number of Time Steps to use',default=10)
    parser.add_argument("--walk_length",dest='walk_length',type=int,help="Length of Walk",default=30)
    parser.add_argument("--num_walks",dest='num_walks',type=int,help="Number of Walks",default=200)
    parser.add_argument("--window", dest='window',type=int,help='Window for word2vec',default=10)
    parser.add_argument("--min_count",dest='min_count',type=int,help='min count',default=1)
    parser.add_argument("--batch_words",dest='batch_words',type=int,help='batch words for word2vec',default=4)   
    parser.add_argument("--embedding_type",dest='embedding_type',type=str,help='Type of Embedding',default='Hadamard')
    parser.add_argument("--threshold_percentage",dest="threshold_percentage",type=float,help='Specifypercentage of weights to remain after threshold',default=0.25)
    parser.add_argument("--threshold_technique",dest="threshold_technique", type=str,help='Per Layer, Global, Percentage removal',default="Global")
    parser.add_argument("--activation_graph",dest="activation_graph",type=str2bool,help='Activation vs. Parameter graph',default=False)
    parser.add_argument("--single_class",dest="single_class",type=int,help='Class Num of Original Network',default=-1)
    args = parser.parse_args()
    print("=> Input Path Prefix: {}".format(args.input_path_prefix))
    print("=> Embedding Save Path: {}".format(args.embedding_save_path))
    print("=> Threshold Percentage: {}".format(args.threshold_percentage))
    print("=> Threshold Technique: {}".format(args.threshold_technique))
    print("=> Single Class Number: {}".format(args.single_class))
    return args

def regex_ctdne(name):
    #we need to pull out relevant experiment information
    #namely, the length of the experiment if we want to align
    #same length runs together
    return
def create_consolidated_graph(parser):
    if parser.activation_graph:
        if parser.single_class > -1: 
            ckpt_path = os.path.join(parser.input_path_prefix, "nx_graphs_activations_{}/".format(parser.single_class))
            print("=> Using Single Class Ckpt Path!")
        else:
            ckpt_path = os.path.join(parser.input_path_prefix, "nx_graphs_activations/")

    else:
        ckpt_path = os.path.join(parser.input_path_prefix, "nx_graphs/")
    print("=> CKPT_PATH: ",ckpt_path)
    ckpt_files = glob.glob(ckpt_path+"*.pkl")
    ckpt_files = sort_nicely(ckpt_files)
    print("=> NUM CKPT FILES: ",len(ckpt_files))
    print("=> Activation Graph?: ",parser.activation_graph)
    assert len(ckpt_files) > 0

    loaded_graphs = []
    for i in tqdm.tqdm(range(parser.time_steps)):
        g_file_name = ckpt_files[i]
        with open(g_file_name,'rb') as file:
            g = pickle.load(file) 
            num_edges = g.number_of_edges()
            if parser.threshold_technique == 'Global':
                #take the top 25% of highest weighted edges
                lower_bound = np.rint(parser.threshold_percentage  *
                                      num_edges).astype(int)
                #sorted in ascending order
                remove = sorted(g.edges(data=True), key=lambda x:np.abs(x[2]['weight']))[:-lower_bound]
                g.remove_edges_from(remove)

            #add back the removed edges nodes!! 
                
            loaded_graphs.append(nx.to_undirected(g))
            nx.set_edge_attributes(loaded_graphs[-1],i,'time') #give edge the same time point

    #convert the loaded graphs to a single multi-edge graph
    consolidated = nx.MultiGraph()
    consolidated.add_nodes_from(loaded_graphs[0]) #all the graphs have the same nodes. 
    #add self loops
    self_loop_list = [(xx,xx)for xx in loaded_graphs[0].nodes()]
    consolidated.add_edges_from(self_loop_list,weight=0,time=0)

    for i in tqdm.tqdm(loaded_graphs):
        consolidated.add_edges_from(i.edges(data=True)) #add all the time stamped edges with weights

    print("Num graphs -- {a} -- Num Nodes -- {b} -- Num Edges -- {c} -- Num Single Edges {d}".format(a=len(loaded_graphs),
       b=consolidated.number_of_nodes(),
       c=consolidated.number_of_edges(),
       d=loaded_graphs[-1].number_of_edges()))
    assert consolidated.number_of_edges() == len(loaded_graphs) * loaded_graphs[-1].number_of_edges() + loaded_graphs[-1].number_of_nodes()

    return consolidated 

def main():
    parser = arg_parser()
    consolidated = create_consolidated_graph(parser)
    exp_path = Path(parser.input_path_prefix).stem

    
    try:
        if parser.activation_graph:
            if parser.single_class > -1:
                embedding_folder = os.path.join(parser.embedding_save_path, exp_path+"_activations_{}".format(parser.single_class))
            else:
                embedding_folder = os.path.join(parser.embedding_save_path, exp_path+"_activations")
        else:
            embedding_folder = os.path.join(parser.embedding_save_path, exp_path)
        os.mkdir(embedding_folder)
        print("Made embedding folder -- {}".format(str(embedding_folder)))
    except OSError as exc:
        print("Embedding folder already exists -- {}".format(str(embedding_folder)))
        if exc.errno != errno.EEXIST:
            print("Other error detected!")
            raise 

    graph_pkl_name = exp_path + "_{}_{}_{}_{}_{}_consolidated_graph.pkl".format(parser.time_steps,parser.walk_length,parser.num_walks, parser.window, parser.batch_words) 
    graph_pkl_location = os.path.join(embedding_folder,graph_pkl_name)
    nx.write_gpickle(consolidated,graph_pkl_location)
    print("=> Saved Consolidated Graph to: ",graph_pkl_location)
    '''
    Create the embeddings
    '''

    CTDNE_model = CTDNE.CTDNE(consolidated,dimensions=parser.dimensions,walk_length=parser.walk_length,num_walks=parser.num_walks,workers=parser.workers)


    model = CTDNE_model.fit(window=parser.window,min_count=parser.min_count,batch_words=parser.batch_words)

   
                        
    exp_prob_name = exp_path + "_{}_{}_{}_{}_{}.probs".format(parser.time_steps,parser.walk_length,parser.num_walks, parser.window, parser.batch_words) 
    prob_location = os.path.join(embedding_folder,exp_prob_name)
  
    try:
        with open(str(prob_location),'wb') as file:
            pickle.dump(model.d_graph,file)
    except:
        print("NO D_GRAPH")

    exp_walk_name = exp_path + "_{}_{}_{}_{}_{}.walks".format(parser.time_steps,parser.walk_length,parser.num_walks, parser.window, parser.batch_words) 
    walk_location = os.path.join(embedding_folder,exp_walk_name)
  
    try:
        with open(str(walk_location),'wb') as file:
            pickle.dump(model.walks,file)
    except:
        print("NO WALKS")
    exp_embedding_name = exp_path + "_{}_{}_{}_{}_{}.embeddings".format(parser.time_steps,parser.walk_length,parser.num_walks, parser.window, parser.batch_words) 
    embedding_location = os.path.join(embedding_folder,exp_embedding_name)
    
    model.wv.save_word2vec_format(str(embedding_location))
    print("word2vec embeddings saved -- {}".format(str(embedding_location)))
     
    model_name = exp_path + "_{}_{}_{}_{}_{}_{}.pth".format(parser.time_steps,parser.walk_length,parser.num_walks, parser.window, parser.batch_words, parser.embedding_type) 
    model_location = os.path.join(embedding_folder,model_name)

    model.save(str(model_location))
    print("model saved -- {}".format(str(model_location)))
    
    "fixme add edge embeddings if necessary."

if __name__ == "__main__":
    main() 
