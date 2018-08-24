import numpy as np
import os
import sys
import configparser
import shutil
from opt_utils import *

class bcolors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

config = configparser.ConfigParser()
config.read('./config.ini')

# MODEL PATH
original_prototxt_path = str(config['DEFAULT']['original_prototxt_path'])
original_model_path = str(config['DEFAULT']['original_model_path'])
optimized_prototxt_path = str(config['DEFAULT']['optimized_prototxt_path'])
optimized_model_path = str(config['DEFAULT']['optimized_model_path'])
mean_proto_path = str(config['DEFAULT']['mean_proto_path'])

# MERGE BN SCALE SETTING
affine_opt = config.getboolean('DEFAULT', 'merge_bn_scale')

# MEAN FILE SETTING
input_scale_opt = config.getboolean('DEFAULT', 'merge_input_scale')
input_mean_opt = config.getboolean('DEFAULT', 'merge_input_mean')
if input_scale_opt:
    input_scale = float(config['DEFAULT']['input_scale'])
if input_mean_opt:
    H = int(config['DEFAULT']['H'])
    W = int(config['DEFAULT']['W'])
    C = int(config['DEFAULT']['C'])
    if os.path.isfile(mean_proto_path):
        mean_vector = np.load(mean_proto_path).mean(1).mean(1)
    else:
        mean_R = float(config['DEFAULT']['mean_R'])
        mean_G = float(config['DEFAULT']['mean_G'])
        mean_B = float(config['DEFAULT']['mean_B'])
        mean_vector = np.array([mean_B, mean_G, mean_R])

# MEMORY OPT SETTING
memory_opt = config.getboolean('DEFAULT', 'merge_inplace_memory')

# ELIMINATE DROPOUT
dropout_opt = config.getboolean('DEFAULT', 'merge_dropout')

# ELIMINATE RELU
relu_opt = config.getboolean('DEFAULT', 'merge_relu')

# MAIN FUNCTION
if memory_opt:
    print "MEMORY In-PLACE OPT : OPEN. IN-PLACE operations reuse the same blob."
    Memo_OPT_Inplace_Memory(original_prototxt_path, original_model_path, optimized_prototxt_path)
    original_prototxt_path = optimized_prototxt_path
else:
    print "MEMORY In-PLACE OPT : CLOSED."
    print bcolors.WARNING + "MEMORY In-PLACE OPT : ** WARNING ** If you want to merge BN and Scale without In-PLACE OPT, please make sure [top.name==bottom.name] in BN and Scale layers." + bcolors.ENDC

if input_mean_opt:
    print "INPUT PREPROCESS (SUB MEAN) OPT : OPEN. MERGE MEAN to the first layer."
    Inpt_OPT_New_Bias(original_prototxt_path, original_model_path, optimized_prototxt_path, optimized_model_path, mean_vector, input_scale, H, W, C)
    original_prototxt_path = optimized_prototxt_path
    original_model_path = optimized_model_path
else:
    print "INPUT PREPROCESS (SUB MEAN) OPT : CLOSED."

if input_scale_opt:
    print "INPUT PREPROCESS (SCALE) OPT : OPEN. MERGE SCALE to the first layer."
    Inpt_OPT_New_Weight(original_prototxt_path, original_model_path, optimized_prototxt_path, optimized_model_path, input_scale)
    original_model_path = optimized_model_path
    original_prototxt_path = optimized_prototxt_path
else:
    print "INPUT PREPROCESS (SCALE) OPT : CLOSED."

if affine_opt:
    print "BN SCALE OPT : OPEN. MERGE BN and SCALE to CONV/IP layers."
    shutil.copyfile(original_prototxt_path, './tmpfile.prototxt')
    AFFine_OPT_Create_Prototxt(original_prototxt_path, optimized_prototxt_path)
    AFFine_OPT_Create_Caffemodel('./tmpfile.prototxt', original_model_path, optimized_prototxt_path, optimized_model_path)
    os.remove('./tmpfile.prototxt')
    original_prototxt_path = optimized_prototxt_path
    original_model_path = optimized_model_path
else:
    print "BN SCALE OPT : CLOSED."

if dropout_opt:
    print "DROPOUT OPT : OPEN. ELIMINATE DROPOUT."
    DrpOut_OPT_Create_Prototxt(original_prototxt_path, original_model_path, optimized_prototxt_path)
    original_prototxt_path = optimized_prototxt_path
    original_model_path = optimized_model_path
else:
    print "DROPOUT OPT : CLOSED."  

if relu_opt:
    print "RELU OPT : OPEN. ELIMINATE RELU."
    ReLU_OPT_Create_Prototxt(original_prototxt_path, original_model_path, optimized_prototxt_path)
else:
    print "RELU OPT : CLOSED." 
