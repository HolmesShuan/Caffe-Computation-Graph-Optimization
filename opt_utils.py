import os
os.environ['GLOG_minloglevel'] = '2'
from google.protobuf.text_format import Parse, MessageToString
import sys
import configparser
import shutil
import numpy as np
# from termcolor import colored

config = configparser.ConfigParser()
config.read('./config.ini', 'utf-8')
# CAFFE PATH
pycaffe_path = config['DEFAULT']['pycaffe_path']
sys.path.append(pycaffe_path)

import caffe
from caffe.proto import  caffe_pb2

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def Memo_OPT_Inplace_Memory(original_prototxt_path, original_model_path, optimized_prototxt_path):
    inplace_operation_type = ['Scale', 'BatchNorm', 'ReLU', 'PReLU', 'Softmax', 'TanH', 'ELU', 'Dropout']
    net_param = caffe_pb2.NetParameter()
    with open(original_prototxt_path, 'rt') as f:
        Parse(f.read(), net_param)
    layer_num = len(net_param.layer)
    parameter_blob_name = []
    blob_pair = {}
    for layer_idx in range(0, layer_num):
        layer = net_param.layer[layer_idx]
        if layer.type in inplace_operation_type and len(layer.bottom)==1:
            if layer.bottom[0] in parameter_blob_name:
                if layer.bottom[0] != layer.top[0]:
                    # inplace opt
                    blob_pair[layer.top[0]] = layer.bottom[0]
                    print "MEMORY In-PLACE OPT : " + layer.name + " : Top Blob [" + layer.top[0] + "] => [" + layer.bottom[0] + "]"
                    net_param.layer[layer_idx].top[0] = layer.bottom[0]
                else:
                    # optimized
                    continue
            else:
                if blob_pair.has_key(layer.bottom[0]):
                    # change bottom blob name
                    blob_pair[layer.top[0]] = blob_pair[layer.bottom[0]]
                    print "MEMORY In-PLACE OPT : " + layer.name + " : Top Blob [" + layer.top[0] + "] => [" + blob_pair[layer.bottom[0]] + "]"
                    print "MEMORY In-PLACE OPT : " + layer.name + " : Bottom Blob [" + layer.bottom[0] + "] => [" + blob_pair[layer.bottom[0]] + "]"
                    net_param.layer[layer_idx].top[0] = blob_pair[layer.bottom[0]]
                    net_param.layer[layer_idx].bottom[0] = blob_pair[layer.bottom[0]]
                else:
                    assert(1>2),"MEMORY In-PLACE OPT : **ERROR** Should Not Reach Here. ##"
        else:
            for i in range(0,len(layer.top)):
                parameter_blob_name.append(layer.top[i])
            for i in range(0,len(layer.bottom)):
                if blob_pair.has_key(layer.bottom[i]):
                    print "MEMORY In-PLACE OPT : " + layer.name + " : Bottom Blob [" + layer.bottom[i] + "] => [" + blob_pair[layer.bottom[i]] + "]"
                    net_param.layer[layer_idx].bottom[i] = blob_pair[layer.bottom[i]]
                else:
                    continue
    with open(optimized_prototxt_path, 'wt') as f:
        f.write(MessageToString(net_param))
    # shutil.copyfile(original_model_path, optimized_model_path)
    print "MEMORY In-PLACE OPT : In-place Memory Optimization Done."
    print bcolors.OKGREEN + "MEMORY In-PLACE OPT : Model at " + original_model_path + "." + bcolors.ENDC
    print bcolors.OKGREEN + "MEMORY In-PLACE OPT : Prototxt at " + optimized_prototxt_path + "." + bcolors.ENDC


def Inpt_OPT_New_Bias(original_prototxt_path, original_model_path, optimized_prototxt_path, new_model_path, mean_vector, scale, H, W, input_channel):
    net_param = caffe_pb2.NetParameter()
    with open(original_prototxt_path, 'rt') as f:
        Parse(f.read(), net_param)
    layer_num = len(net_param.layer)

    new_net_param = caffe_pb2.NetParameter()
    new_net_param.name = 'calc_new_bias'
    new_net_param.layer.add()
    new_net_param.layer[-1].name = "data"
    new_net_param.layer[-1].type = 'Input'
    new_net_param.layer[-1].top.append('data')
    new_net_param.layer[-1].input_param.shape.add()
    new_net_param.layer[-1].input_param.shape[-1].dim.append(1)
    new_net_param.layer[-1].input_param.shape[-1].dim.append(int(input_channel))
    new_net_param.layer[-1].input_param.shape[-1].dim.append(int(H))
    new_net_param.layer[-1].input_param.shape[-1].dim.append(int(W))

    target_blob_name = ''
    target_layer_name = ''
    input_layer_type = ['Data', 'Input', 'AnnotatedData']
    for layer_idx in range(0, layer_num):
        layer = net_param.layer[layer_idx]
        if layer.type not in input_layer_type:
            assert(layer.type=='Convolution' or layer.type=='InnerProduct'), "## ERROR : First Layer MUST BE CONV or IP. ##"
            new_net_param.layer.extend([layer])
            if layer.type=='Convolution':
                try:
                    assert(new_net_param.layer[-1].convolution_param.pad[0] == 0), '## ERROR : MEAN cannot be mearged into CONV with padding > 0. ##'
                except:
                    # padding not set
                    pass
                target_blob_name = layer.top[0]
                target_layer_name = layer.name
            break

    new_proto_name = './tmpfile.prototxt'
    with open(new_proto_name, 'wt') as f:
        f.write(MessageToString(new_net_param))
    caffe.set_mode_cpu()
    net = caffe.Net(new_proto_name, str(original_model_path), caffe.TEST)

    mean_array = mean_vector*(-1.0)*scale
    mean_array = mean_array.reshape(input_channel, 1)
    mean_array = np.tile(mean_array, (1,H*W)).reshape(1, input_channel, H, W)

    os.remove(new_proto_name)
     
    net.blobs['data'].data[...] = mean_array
    net.forward()
    mean_data = net.blobs[target_blob_name].data[...]
    mean_data = mean_data.reshape(mean_data.shape[1],mean_data.shape[2]*mean_data.shape[3])
    new_bias = np.mean(mean_data,1) 
    print "INPUT PREPROCESS (SUB MEAN) OPT : Calc New Bias Done."

    caffe.set_mode_cpu()
    net = caffe.Net(original_prototxt_path, str(original_model_path), caffe.TEST)
    if len(net.params[target_layer_name]) == 2:
        # with bias
        net.params[target_layer_name][1].data[...] += new_bias[...]
        net.save(new_model_path)
        try:
            shutil.copyfile(original_prototxt_path, optimized_prototxt_path)
        except:
            # same file, not need to copy
            pass
        print "INPUT PREPROCESS (SUB MEAN) OPT : Merge Mean Done."
        print bcolors.OKGREEN + "INPUT PREPROCESS (SUB MEAN) OPT : Model at " + new_model_path + "." + bcolors.ENDC
        print bcolors.OKGREEN + "INPUT PREPROCESS (SUB MEAN) OPT : Prototxt at " + optimized_prototxt_path + "." + bcolors.ENDC
        print bcolors.WARNING + "INPUT PREPROCESS (SUB MEAN) OPT : ** WARNING ** Remember to set mean values to zero before test !!!" + bcolors.ENDC

    else:
        net_param = caffe_pb2.NetParameter()
        with open(original_prototxt_path, 'rt') as f:
            Parse(f.read(), net_param)
        layer_num = len(net_param.layer)
        for layer_idx in range(0, layer_num):
            layer = net_param.layer[layer_idx]
            if layer.name == target_layer_name:
                if layer.type == 'Convolution':
                    net_param.layer[layer_idx].convolution_param.bias_term = True
                else:
                    net_param.layer[layer_idx].inner_product_param.bias_term = True
                break
        with open(optimized_prototxt_path, 'wt') as f:
            f.write(MessageToString(net_param))

        new_net = caffe.Net(optimized_prototxt_path, caffe.TEST)
        for param_name in net.params.keys():
            for i in range(0,len(net.params[param_name])):
                new_net.params[param_name][i].data[...] = net.params[param_name][i].data[...]
        new_net.params[target_layer_name][1].data[...] = new_bias[...]
        new_net.save(new_model_path)
        print "INPUT PREPROCESS (SUB MEAN) OPT : Merge Mean Done."
        print bcolors.OKGREEN + "INPUT PREPROCESS (SUB MEAN) OPT : Model at " + new_model_path + "." + bcolors.ENDC
        print bcolors.OKGREEN + "INPUT PREPROCESS (SUB MEAN) OPT : Prototxt at " + optimized_prototxt_path + "." + bcolors.ENDC
        print bcolors.WARNING + "INPUT PREPROCESS (SUB MEAN) OPT : ** WARNING ** Remember to set mean values to zero before test !!!" + bcolors.ENDC

def Inpt_OPT_New_Weight(original_prototxt_path, original_model_path, optimized_prototxt_path, new_model_path, scale):
    net_param = caffe_pb2.NetParameter()
    with open(original_prototxt_path, 'rt') as f:
        Parse(f.read(), net_param)
    layer_num = len(net_param.layer)
    input_layer_type = ['Data', 'Input', 'AnnotatedData']
    for layer_idx in range(0, layer_num):
        layer = net_param.layer[layer_idx]
        if layer.type not in input_layer_type:
            assert(layer.type=='Convolution' or layer.type=='InnerProduct'), "## ERROR : First Layer MUST BE CONV or IP. ##"
            target_layer_name = layer.name
            break
        else:
            try:
                net_param.layer[layer_idx].transform_param.scale = 1.0
            except:
                print bcolors.WARNING + "INPUT PREPROCESS (SCALE) OPT : ** WARNING ** NO SCALE found in DATA layer." + bcolors.ENDC

    new_net = caffe.Net(original_prototxt_path, str(original_model_path), caffe.TEST)
    new_net.params[target_layer_name][0].data[...] = new_net.params[target_layer_name][0].data[...]*scale
    new_net.save(new_model_path)

    with open(optimized_prototxt_path, 'wt') as f:
        f.write(MessageToString(net_param))

    print "INPUT PREPROCESS (SCALE) OPT : Merge Input Scale Done."
    print bcolors.OKGREEN + "INPUT PREPROCESS (SCALE) OPT : Model at " + new_model_path + "." + bcolors.ENDC
    print bcolors.OKGREEN + "INPUT PREPROCESS (SCALE) OPT : Prototxt at " + optimized_prototxt_path + "." + bcolors.ENDC
    #print "INPUT PREPROCESS (SCALE) OPT : ## TIPS ## Remember to remove scale in data layer before test !!!"

def AFFine_OPT_Create_Prototxt(original_prototxt_path, optimized_prototxt_path):
    net_param = caffe_pb2.NetParameter()
    new_net_param = caffe_pb2.NetParameter()
    with open(original_prototxt_path, 'rt') as f:
        Parse(f.read(), net_param)
    layer_num = len(net_param.layer)

    parameter_layer_type = ['Convolution', 'InnerProduct']
    merge_layer_type = ['Scale', 'BatchNorm']

    for layer_idx in range(0, layer_num):
        layer = net_param.layer[layer_idx]
        if layer.type not in merge_layer_type:
            new_net_param.layer.extend([layer])
        else:
            if layer.type == 'Scale' and len(layer.bottom) != 1:
                # In case, scale layer has two bottom blob, then scale layer can't be merged into CONV/IP.
                new_net_param.layer.extend([layer])
            else:
                continue
        if layer.type in parameter_layer_type:
            if layer_idx+1 < layer_num:
                if net_param.layer[layer_idx+1].type in merge_layer_type and len(net_param.layer[layer_idx+1].bottom)==1:
                    # In case, scale layer has two bottom blob, then scale layer can't be merged into CONV/IP.
                    if layer.type == 'Convolution':
                        new_net_param.layer[-1].convolution_param.bias_term = True
                    else:
                        new_net_param.layer[-1].inner_product_param.bias_term = True
    new_net_param.name = net_param.name
    with open(optimized_prototxt_path, 'wt') as f:
        f.write(MessageToString(new_net_param))
    print "BN SCALE OPT : Create Optimized Prototxt Done."
    print bcolors.OKGREEN + "BN SCALE OPT : Prototxt at " + optimized_prototxt_path + "." + bcolors.ENDC

def AFFine_OPT_Create_Caffemodel(original_prototxt_path, original_model_path, optimized_prototxt_path, new_model_path):
    net_param = caffe_pb2.NetParameter()
    with open(original_prototxt_path, 'rt') as f:
        Parse(f.read(), net_param)

    param_layer_type_list = [layer.type for layer in net_param.layer]
    param_layer_name_list = [layer.name for layer in net_param.layer]
    target_layer_type = ['Convolution', 'InnerProduct']
    merge_layer_type = ['Scale', 'BatchNorm']

    caffe.set_mode_cpu()
    net = caffe.Net(original_prototxt_path, original_model_path, caffe.TEST)
    new_net = caffe.Net(optimized_prototxt_path, caffe.TEST)
    for param_name in new_net.params.keys():
        param_layer_idx = param_layer_name_list.index(param_name)
        param_layer_type = param_layer_type_list[param_layer_idx]
        if param_layer_type not in target_layer_type:
            # OTHER LAYERS
            for i in range(0,len(net.params[param_name])):
                new_net.params[param_name][i].data[...] = net.params[param_name][i].data[...]
        else:
            kernel_num = net.params[param_name][0].num
            new_net.params[param_name][0].data[...] = net.params[param_name][0].data[...]
            if len(net.params[param_name]) == 2:
                new_net.params[param_name][1].data[...] = net.params[param_name][1].data[...]
            #else:
            #    print new_net.params[param_name][1].data[...]
            if param_layer_idx+1 < len(param_layer_type_list):
                for i in range(param_layer_idx+1, len(param_layer_type_list)):
                    # CHECK : CONV + BN +SCALE / CONV + BN / IP + ... 
                    affine_layer_type = param_layer_type_list[i]
                    affine_layer_name = param_layer_name_list[i]
                    if affine_layer_type in merge_layer_type:
                        # MERGE BN/SCALE
                        if affine_layer_type == "Scale":
                            if len(net_param.layer[i].bottom)>=2:
                                # NOT In-place Scale
                                try:
                                    for j in range(0,len(net.params[affine_layer_name])):
                                        new_net.params[affine_layer_name][j].data[...] = net.params[affine_layer_name][j].data[...]
                                except:
                                    # no parameter
                                    break
                            else:
                                # In-place Scale
                                scale = net.params[affine_layer_name][0].data
                                if len(net.params[affine_layer_name]) == 2:
                                    bias = net.params[affine_layer_name][1].data
                                else:
                                    bias = 0.0*scale
                                for k in range(0, kernel_num):
                                    new_net.params[param_name][0].data[k] = new_net.params[param_name][0].data[k]*scale[k]
                                    new_net.params[param_name][1].data[k] = new_net.params[param_name][1].data[k]*scale[k]+bias[k]
                        elif affine_layer_type == "BatchNorm":
                            scale = net.params[affine_layer_name][2].data[0]
                            if scale != 0:
                                mean = net.params[affine_layer_name][0].data / scale
                                std = np.sqrt(net.params[affine_layer_name][1].data / scale)
                            else:
                                mean = net.params[affine_layer_name][0].data
                                std = np.sqrt(net.params[affine_layer_name][1].data)
                            for k in range(0, kernel_num):
                                new_net.params[param_name][0].data[k] = new_net.params[param_name][0].data[k] / std[k]
                                new_net.params[param_name][1].data[k] = (new_net.params[param_name][1].data[k] - mean[k]) / std[k]
                        else:
                            # TODO
                            assert(1>2), "## TODO ## : Other layers haven't been supported yet. ##"
                    else:
                        # NOT BN or SCALE, then BREAK
                        break
            else:
                # LAST LAYER, then BREAK
                break
    new_net.save(new_model_path)    
    print bcolors.OKGREEN + "BN SCALE OPT : Model at " + new_model_path + "." + bcolors.ENDC

def DrpOut_OPT_Create_Prototxt(original_prototxt_path, original_model_path, optimized_prototxt_path):
    net_param = caffe_pb2.NetParameter()
    new_net_param = caffe_pb2.NetParameter()
    with open(original_prototxt_path, 'rt') as f:
        Parse(f.read(), net_param)
    for layer_idx in range(0, len(net_param.layer)):
        layer = net_param.layer[layer_idx]
        if layer.type == 'Dropout':
            if layer.top[0] == layer.bottom[0]:
                continue
            else:
                new_net_param.layer[-1].top[0] = layer.top[0]
        else:
            new_net_param.layer.extend([layer])
    new_net_param.name = net_param.name
    with open(optimized_prototxt_path, 'wt') as f:
        f.write(MessageToString(new_net_param))
    print "DROPOUT OPT : Create Optimized Prototxt Done."
    print bcolors.OKGREEN + "DROPOUT OPT : Model at " + original_model_path + "." + bcolors.ENDC
    print bcolors.OKGREEN + "DROPOUT OPT : Prototxt at " + optimized_prototxt_path + "." + bcolors.ENDC

