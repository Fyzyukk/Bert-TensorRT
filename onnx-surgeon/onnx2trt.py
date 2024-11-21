# -*- coding: utf-8 -*-
import ctypes

import tensorrt as trt
import os
from glob import glob




def onnx2trt(onnxFile, plan_name):
    #1初始化一个logger日志记录器
    Layer_norm_plugin_path  = r'D:\LayerNormPlugin\cmake-build-debug\LayerNormPlugin.dll'
    ctypes.cdll.LoadLibrary(Layer_norm_plugin_path)

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
    #network.get_input(n)：获取网络的输入节点，这
    inputTesnor = network.get_input(0)
    inputlengths = network.get_input(1)
    #profile.set_shape：为每个输入设置优化配置文件中不同的形状范围。
    profile.set_shape(inputTesnor.name, (1, 16,80), (16, 64,80), (64, 256,80))
    profile.set_shape(inputlengths.name, [1],[16],[64])
    config.add_optimization_profile(profile)#将优化配置添加到config

    # config.flags = config.flags & ~(1 << int(trt.BuilderFlag.FP32))
    # config.set_flag(trt.BuilderFlag.STRICT_TYPES)


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



if __name__ == '__main__':
    src_onnx = './encoder_v1.onnx'
    plan_name = './encoder_V1.plan'
    onnx2trt(src_onnx,plan_name)
