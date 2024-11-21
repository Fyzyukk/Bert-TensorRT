import tensorrt as trt
import os

import numpy as np
import onnx

import torch
from torch.nn import functional as F
import numpy as np
import os
from transformers import BertTokenizer, BertForMaskedLM

# import onnxruntime as ort
import transformers

from trt_helper import *

def onnx2trt(onnxFile, plan_name):

    #1初始化一个logger日志记录器
    logger = trt.Logger(trt.Logger.VERBOSE)

    #2tensorrt的构建和配置
    builder = trt.Builder(logger)  #构建trt.Builder对象,日志记录器
    config = builder.create_builder_config()   #config接受Builder对象的配置，例如最大工作空间，精度等 c++ builer->createBUilderConfig()
    profile = builder.create_optimization_profile()   #profile优化配置文件，定义输入尺寸的动态变化范围，便于于tensorrt根据范围生成更优化的engine
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) #创建一个空的网络定义 C++用builder->createNetwork()
    #trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH显示批量模式，网络二点输入和尺寸需要显示的指定大小，动态设置。网络会明确的在每一层的输入和输出表示批量维度，而不是根据构建是自动推断。
    #1<<x 左移 二进制进行2^x操作，EXPLICIT_BATCH的值是0，int强转，相当于1<<0 =1 表示启用了显示批量模式
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30) #config中的参数，构建engine允许使用的最大工作空间

    #3解析onnx模型
    parser = trt.OnnxParser(network, logger) #trt.OnnxParser将onnx模型转换为tensorrt模型的API接口 需要两个参数，是上面检测network和logger 但是tensorrt不知道网络层之间的连接关系
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")  #读取文件
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()): #通过 parser 对象解析读取的 ONNX 文件数据。model.read() 将文件内容读取为字节串，并传递给 parser
            #parser.parse() 接收一个字节流并尝试解析其中的模型结构，成功则返回 True，失败则返回 False，并可以通过 parser.num_errors 和 parser.get_error() 方法查看错误信息
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
    print("Succeeded parsing ONNX file!")

    #4优化配置文件的输入形状
    #network.get_input(n)：获取网络的输入节点，这里是通过索引获取输入的三个张量（input_ids, token_type_ids, input_mask）
    input_ids = network.get_input(0)
    token_type_ids = network.get_input(1)
    input_mask = network.get_input(2)
    #profile.set_shape：为每个输入设置优化配置文件中不同的形状范围。(1, 6) 是最小形状，(1, 64) 是优化形状，(1, 256) 是最大形状。这些形状会告诉 TensorRT 网络的输入张量可能的维度变化范围，TensorRT 将据此优化推理过程
    profile.set_shape(input_ids.name, (1, 16), (1, 16), (1, 16))
    profile.set_shape(token_type_ids.name, (1, 16), (1, 16), (1, 16))
    profile.set_shape(input_mask.name, (1, 16), (1,16), (1, 16))
    config.add_optimization_profile(profile)#将优化配置添加到config


    #5构建tensorrt的engine
    engine = builder.build_engine_with_config(network, config) #根据network和config配置构建引擎engine
    if not engine:
        raise RuntimeError("build_engine failed")
    print("Succeeded building engine!")

    #6序列号engine保存为plan文件
    print("Serializing Engine...")
    serialized_engine = engine.serialize() #将tensorrrt序列化为字节流  序列化
    if serialized_engine is None:
        raise RuntimeError("serialize failed")

    with open(plan_name, "wb") as fout:
        fout.write(serialized_engine)  #将序列化后的engine保存到指定的目录中

def trt_infer(plan_name,PATH):
    print(os.listdir(PATH))
    tokenizer = BertTokenizer.from_pretrained(PATH)
    #加载了一个 BERT Tokenizer，使用的是预训练的 bert-base-uncased 模型。BERT tokenizer 用于将文本转化为模型可以处理的输入格式，如 token IDs
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower." #构建输入文本
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)
    print(encoded_input) #打印编码后的输入

    """
    TensorRT Initialization
    """
    #1tensorRT初始化
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    #2创建推理助手对象
    infer_helper = InferHelper(plan_name, TRT_LOGGER)
    #输入数据
    input_list = [encoded_input['input_ids'].detach().numpy(), encoded_input['token_type_ids'].detach().numpy(), encoded_input['attention_mask'].detach().numpy()]
    #使用 infer_helper 对输入数据 input_list 进行推理，并得到模型的输出。output 是模型的预测结果
    output = infer_helper.infer(input_list)
    print(output)

    #output[0] 是模型的原始预测结果，即 logits，这些值表示每个词在 mask_token 位置上的概率分布
    logits = torch.from_numpy(output[0])
    #F.softmax 对 logits 进行 softmax 归一化，将其转换为概率分布，表示每个词的预测概率
    softmax = F.softmax(logits, dim = -1)
    #获取掩码位置的概率分布
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    print(top_10)
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)

    print("hhhh, the result is wrong, debug yourself")

if __name__ == '__main__':
    src_onnx = 'D:/Code/TensorRT-main/model-sim.onnx'
    plan_name = 'D:/Code/TensorRT-main/bert.plan'
    BERT_PATH = 'D:/Code/TensorRT-main/BERT'
    #onnx2trt(src_onnx, plan_name)
    trt_infer(plan_name,BERT_PATH)
