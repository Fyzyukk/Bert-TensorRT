#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from jinja2.nodes import Output
from sympy.codegen.fnodes import elemental

# import torch
import tensorrt as trt
import numpy as np
import ctypes
import math
import time

from typing import Optional, Tuple

import pycuda.driver as cuda
import pycuda.autoinit
from scipy.constants import precision
from sympy.core.random import shuffle

######################plugin##############
LayerNorm_plugin_path = 'D:/Code/TensorRT-main/LayerNormPlugin/cmake-build-debug/LayerNormPlugin.dll'
#xx名字_plugin_path = 'xxxxxxxxxxxxxxxxxxxx'
#####################get plugin##########

def getPlugin(name):
    ctypes.cdll.LoadLibrary(globals().get(name + '_plugin_path'))
    for c in trt.get_plugin_registry().plugin_creator_list:
        # print(c.name)
        if c.name == name:
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

class TrtNetworkHelper(): #TrtNetworkHelper 的类，它主要用于简化和辅助 TensorRT 网络构建和操作
    """TensorRT Network Definition helper for Pytorch"""
    def __init__(self, network, plugin_registry, logger):
        self.network = network
        self.plugin_registry = plugin_registry
        self.logger = logger

        self.input_num = 0

    def set_layer_name(self, trt_layer, name): # TODO 为 TensorRT 层（layer）设置名称并打印该层的输出形状
        """
        Tool function. Set the name of trt layer or plugin and print output shapes.
        """
        if not trt_layer:
            raise RuntimeError("Could not name")

        #为传入的 layer 设置名称。这里的名称由网络中当前层的编号（通过 self.network.num_layers 获取）和传入的 name 参数组成
        #self.network.num_layers 返回当前网络中已有的层数，以便为每一层创建唯一的名称，防止重复命名
        trt_layer.name = str(self.network.num_layers) + "_" + name
        for i in range(0, trt_layer.num_outputs):
            #遍历 layer 的输出张量（数量为 layer.num_outputs），获取每个输出的形状信息
            #查看该层有几个输出
            shape = trt_layer.get_output(i).shape
            #调用 layer.get_output(i).shape 获得其形状（shape）
            self.logger.log(trt.Logger.INFO, "[Network] " + trt_layer.name + ", output[" + str(i) + "] shape= " + str(shape))
            #self.logger.log 将每个输出张量的形状信息记录到日志中，方便调试和分析

        return None

    def check_trt_layer(self, trt_layer): # TODO 检查 传入的TensorRT 层是否有效
        """
        Tool function. check trt layer,
        """
        if not trt_layer:
            raise RuntimeError("add " + str(trt_layer) + " failed!")
        #首先判断传入的 trt_layer 是否为空（None）。如果为空，则抛出一个 RuntimeError 异常，提示创建该层失败。这可以帮助快速定位错误，特别是在构建网络过程中，确保每个层都被成功创建

        for i in range(0, trt_layer.num_outputs):
            #遍历该层的输出张量（通过 trt_layer.num_outputs 获取输出数量）
            shape = trt_layer.get_output(i).shape
            #trt_layer.get_output(i).shape 获取其形状信息，并存储在 shape 中

            # print(trt.volume(shape))

            # if len(shape) is 1:
                # raise RuntimeError("add " + layer.name + " failed!")

    def layer_post_process(self, trt_layer, layer_name, precision): # TODO 检查添加的层
        #trt_layer：需要后处理的TensorRT层
        #layer_name：层的名称，用于标识层并帮助调试
        #precision：该层的计算精度（例如 float32、float16 等）
        """
        Tool function. set precision, set_layer_name and check_trt_layer
        """
        if precision is not None:
            trt_layer.precision = precision

        self.set_layer_name(trt_layer, layer_name)
        #调用上面自定义的 set_layer_name 方法，将 layer_name 作为层的标识并打印输出张量的形状信息
        self.check_trt_layer(trt_layer)
        #调用上面自定义的 check_trt_layer 方法，验证 trt_layer 是否成功创建并检查输出张量形状

    def addInput(self, name, dtype, shape): #在 TODO 添加输入
        if name is None:
            name = "input" + str(self.input_num)
            #如果 name 为 None，则生成默认名称，例如 "input0"，依次递增

        self.input_num = self.input_num + 1
        #更新 input_num 计数器，为下一个输入层生成新的默认名称

        trt_input = self.network.add_input(name=name, dtype=dtype, shape=shape)
        #调用 TensorRT 的 add_input 函数，使用指定名称、数据类型和形状在网络中添加一个输入层，并将生成的输入对象赋给 trt_input
        if not trt_input:
            raise RuntimeError("addInput failed!")

        self.logger.log(trt.Logger.INFO, "[Network] add input:" + name + ", shape=" + str(shape))
        #使用 TensorRT 的日志记录器，记录输入层的名称和形状，方便调试和跟踪

        return trt_input # TODO 返回的是创建的输入层对象 是trt.ITensor
        #返回创建的输入层对象 trt_input，供后续操作

    def markOutput(self, x: trt.ITensor): # TODO 添加输出
        self.network.mark_output(x)
        #self.network.mark_output(x) 将张量 x 标记为 TensorRT 网络的输出张量
        self.logger.log(trt.Logger.INFO, "[Network] mark output:" + x.name + ", shape=" + str(x.shape))

    def addEmbedding(self, indices, weight, layer_name=None, precision=None): # TODO 实现了嵌入层（Embedding Layer）的功能，通过 gather 操作从常量权重中根据索引提取值
        constant_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
        #add_constant 用于在网络中添加一个常量张量层，这里传入了嵌入层的权重 weight。生成的 constant_layer 中包含了整个嵌入层的权重矩阵，存储在 GPU 显存中
        gather_layer = self.network.add_gather(constant_layer.get_output(0),indices, axis=0)
        #add_gather 实现了类似于“查表”的操作。gather_layer 将根据输入 indices 从 constant_layer 中按行抽取指定的值，等效于嵌入层中用索引来查找对应的嵌入向量

        if layer_name is None:
            layer_name = "nn.Embedding"
        else:
            layer_name = "nn.Embedding." + layer_name

        self.layer_post_process(gather_layer, layer_name, precision)

        return gather_layer.get_output(0)

    def addGELU(self, x, layer_name=None, precision=None): # TODO 实现了 GELU 激活函数的计算
        POW = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
        #add_constant的操作实际上是创建一个一些常量 大小为1x1x1个weights，
        MULTIPLY = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
        SQRT = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
        ONE = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
        HALF = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
        X_pow = self.network.add_elementwise(x, POW.get_output(0), trt.ElementWiseOperation.POW)
        #add_elementwise逐元素操作
        X_pow_t = X_pow.get_output(0)
        X_mul = self.network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
        X_add = self.network.add_elementwise(x, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
        X_sqrt = self.network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
        X_sqrt_tensor = X_sqrt.get_output(0)
        X_tanh = self.network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
        X_tanh_tensor = X_tanh.get_output(0)
        X_one = self.network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
        CDF = self.network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
        gelu_layer = self.network.add_elementwise(CDF.get_output(0), x, trt.ElementWiseOperation.PROD)

        if layer_name is None:
            layer_name = "nn.GELU"
        else:
            layer_name = "nn.GELU." + layer_name

        self.layer_post_process(gelu_layer, layer_name, precision)

        return gelu_layer.get_output(0)

    def addLayerNorm(self, x, gamma, beta, layer_name=None, precision=None): # TODO 添加一个 LayerNorm 层的 TensorRT 实现
        LayerNrom_Plugin = getPlugin('LayerNorm')

        if LayerNrom_Plugin is None:
            raise RuntimeError("LayerNorm Plugin not found!")

        inputTensorList = []
        inputTensorList.append(x)
        LayerNrom_layer = self.network.add_plugin_v2(inputTensorList, LayerNrom_Plugin)

        # gamma_tensor = self.network.add_constant(gamma.shape,gamma).get_output(0)
        # beta_tensor = self.network.add_constant(beta.shape,beta).get_output(0)

        # gamma_tensor = self.network.add_shuffle(gamma_tensor)
        # gamma_tensor.reshape_dims = (1,1,gamma_tensor.get_output(0).shape[0])
        # beta_tensor = self.network.add_shuffle(beta_tensor)
        # beta_tensor.reshape_dims = (1,1,beta_tensor.get_output(0).shape[0])


        # LayerNrom_layer_ouput = self.network.add_elementwise(LayerNrom_layer.get_output(0),gamma_tensor.get_output(0),trt.ElementWiseOperation.PROD)
        # LayerNrom_layer_ouput = self.network.add_elementwise(LayerNrom_layer_ouput.get_output(0),beta_tensor.get_output(0),trt.ElementWiseOperation.SUM)

        if layer_name is None:
            layer_name = "nn.LayerNorm"
        else:
            layer_name = "nn.LayerNorm." + layer_name

        self.layer_post_process(LayerNrom_layer, layer_name, precision)

        return LayerNrom_layer.get_output(0)

    def addLinear(self, x, weight, bias, layer_name=None, precision=None): # TODO 一个线性层（Fully Connected Layer） y=x*weight+bias
        # 输入张量 x：表示当前层的输入数据
        # 权重 weight：这是一个矩阵，它与输入数据进行乘法运算。权重决定了每个输入特征如何影响输出。
        # 偏置 bias：这是一个向量，它会加到输入与权重乘积的结果中，帮助模型更好地拟合数据。
        # 输出：计算的结果是输入与权重的矩阵乘法再加上偏置
        weight = np.array([weight])
        constant_layer = self.network.add_constant(weight.shape,trt.Weights(weight))
        X_mul = self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)

        bias = np.array([[bias]])
        bias_layer = self.network.add_constant(bias.shape,trt.Weights(bias))

        linear_layer = self.network.add_elementwise(X_mul.get_output(0), bias_layer.get_output(0), trt.ElementWiseOperation.SUM)

        if layer_name is None:
            layer_name = "nn.Linear"
        else:
            layer_name = "nn.Linear." + layer_name

        self.layer_post_process(linear_layer,layer_name,precision)

        return linear_layer.get_output(0)

    def addReLU(self, layer, x, layer_name=None, precision=None) -> trt.ITensor: # TODO 在网络中添加一个 ReLU 激活层
        relu_layer = self.network.add_activation(x, type=trt.ActivationType.RELU)

        if layer_name is None:
            layer_name = "nn.ReLU"

        self.layer_post_process(relu_layer, layer_name, precision)

        return relu_layer.get_output(0)

    def addSoftmax(self, x: trt.ITensor, dim: int = -1, layer_name=None, precision=None) -> trt.ITensor: # TODO 在网络中添加一个 Softmax 层
        softmax_layer = self.network.add_softmax(x)

        input_len = len(x.shape)
        if dim == -1:
            dim = input_len
        softmax_layer.axes = int(math.pow(2,input_len-1))

        layer_name_prefix = "nn.Softmax[dim=" + str(dim) + "]"
        if layer_name is None:
            layer_name = layer_name_prefix
        else:
            layer_name = layer_name_prefix + "." +layer_name

        self.layer_post_process(softmax_layer,layer_name, precision)

        return softmax_layer.get_output(0)

    ################## unary op ###################
    def addLog(self, x: trt.ITensor, layer_name=None, precision=None): # TODO 在网络中添加一个 Log 操作层
        log_layer = self.network.add_unary(x, trt.UnaryOperation.LOG)
        if layer_name is None:
            layer_name = "unary.log"
        else:
            layer_name = "unary.log." + layer_name

        self.layer_post_process(log_layer, layer_name, precision)

        return log_layer.get_output(0)

    ################## elementwise op ###################
    def addAdd(self, a:trt.ITensor, b:trt.ITensor, layer_name=None, precision=None): # TODO 在网络中添加一个加法层
        add_layer = self.network.add_elementwise(a,b,trt.ElementWiseOperation.SUM)

        if layer_name:
            add_layer.name = layer_name
        if precision:
            add_layer.precision = precision

        return add_layer.get_output(0)


    # tensor and scalar op
    def addScale(self,x: trt.ITensor,scale: float,layer_name: str = None,precision: trt.DataType = None) -> trt.ITensor: # TODO 添加一个缩放操作
        """scale"""
        input_len = len(x.shape)
        if input_len < 3:
            raise RuntimeError ("input_len <3 not support now! ")
        if layer_name is None:
            layer_name = "scale"

        #the input demension must be greater than or equal to 4
        if input_len == 3:
            scale_layer = self.network.add_shuffle(x)
            #当 x 为 3 维时，使用 add_shuffle 层将其扩展为 4 维。(0, 0, 0, 1) 的含义是保持前 3 维不变，并在末尾添加一个维度，使其成为形状为 (d1, d2, d3, 1) 的 4 维张量
            scale_layer.reshape_dims = (0,0,0,1)
            self.layer_post_process(scale_layer, layer_name+".3dto4d", precision)

        np_scale = trt.Weights(np.array([scale],dtype = np.float32))
        scale_layer = self.network.add_scale(x, trt.ScaleMode.UNIFORM, None, np_scale, None)
        self.layer_post_process(scale_layer,layer_name,precision)

        if input_len == 3:
            scale_layer = self.network.add_shuffle(x)
            #如果输入是 3 维（经过扩展为 4 维后），则使用 add_shuffle 层将其恢复到原始的 3 维形状。
            scale_layer.reshape_dims = (0,0,0)
            self.layer_post_process(scale_layer,layer_name+".4dto3d",precision)

        return scale_layer.get_output(0)

    def addMatMul(self, a: trt.ITensor, b: trt.ITensor, layer_name: Optional[str] = None) -> trt.ITensor: # TODO 添加矩阵乘法操作（MatMul）
        matmul_layer = self.network.add_matrix_multiply(a, trt.MatrixOperation.NONE, b, trt.MatrixOperation.NONE)
        if layer_name is None:
            layer_name = "matmul"
        else:
            layer_name = "matmul." + layer_name

        self.layer_post_process(matmul_layer,layer_name,None)

        return matmul_layer.get_output(0)


    def addConstant(self, w, layer_name: Optional[str] = None) -> trt.ITensor: # TODO 在网络中添加一个常量层
        constant_layer = self.network.add_constant(w.shape, w)

        if layer_name is None:
            layer_name = "trt.Constant"
        else:
            layer_name = "trt.Constant." + layer_name

        self.layer_post_process(constant_layer, layer_name, None)

        return constant_layer

    def addShuffle(self,x: trt.ITensor,
                   first_transpose: trt.Permutation, # TODO 进行转置或者变换维度操作
                   reshape_dims: trt.Dims,
                   second_transpose: trt.Permutation,
                   layer_name: Optional[str] = None) -> trt.ITensor:
        """"""
        shuffle_layer = self.network.add_shuffle(x)
        if first_transpose is not None:
            shuffle_layer.first_transpose = first_transpose

        if reshape_dims is not None:
            shuffle_layer.reshape_dims = reshape_dims

        if second_transpose is not None:
            shuffle_layer.second_transpose = second_transpose

        if layer_name is None:
            layer_name = "trt.Shuffle"
        else:
            layer_name = "trt.Shuffle." + layer_name

        self.layer_post_process(shuffle_layer, layer_name, None)

        return shuffle_layer.get_output(0)


class InferHelper(): # TODO InferHelper 类的实现，主要用于执行 TensorRT 模型推理
    """"""
    def __init__(self, plan_name, trt_logger):
        """"""
        self.logger = trt_logger
        self.runtime = trt.Runtime(trt_logger)
        with open(plan_name, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            if not self.engine:
                print("Failed to deserialize engine!")
            else:
                print("Engine deserialized successfully.")
            self.context = self.engine.create_execution_context()
            # self.context.active_optimization_profile = 0

    def infer(self, inputs: list): # TODO infer：这是用于执行推理的函数，输入是一个包含多个张量（输入数据）的列表
        nInput = len(inputs)

        bufferD = [] #TODO 创建一个空的列表 bufferD，用于存储输入张量在设备上的内存地址
        # alloc memory
        for i in range(nInput): #TODO 为每个输入张量申请GPU空间
            bufferD.append(cuda.mem_alloc_like(inputs[i]))
            # cuda.memcpy_htod(bufferD[i], inputs[i].ravel())
            bufferD[i] = cuda.to_device(inputs[i]) # TODO 将输入张量数据从CPU复制到GPU
            # self.context.set_binding_shape(i, tuple(inputs[i].shape))
            # TODO python接口的trt10.6版本中这个绑定已经删除了，使用set_input_shape
            tensor_name = self.engine.get_tensor_name(i)
            print(f"Setting shape for tensor: {tensor_name}, shape: {inputs[i].shape}")
            self.context.set_input_shape(tensor_name, tuple(inputs[i].shape))
        print("Binding all? %s"%(["No","Yes"][int(self.context.all_binding_shapes_specified)]))

        # for i in range(0, self.engine.num_bindings):
        #     print("get_binding_shape:" + str(self.context.get_tensor_shape(i)))
        # vocab_size = 30522
        outputs = [] #  TODO 创建输出缓冲区
        for i in range(nInput,4):
            tensor_name = self.engine.get_tensor_name(i)
            output_shape = self.context.get_tensor_shape(tensor_name)  # 获取输出张量的形状
            print(f"Setting up output tensor {tensor_name} with shape {output_shape}")
            outputs.append(np.zeros(self.context.get_tensor_shape(tensor_name),dtype = np.float32))
            # output_shape = self.context.get_binding_shape(i)

        nOutput = len(outputs) # TODO 在GPU上为输出申请空间
        for i in range(nOutput):
            # bufferD.append(cuda.mem_alloc(outputs[i].nbytes))
            bufferD.append(cuda.mem_alloc_like(outputs[i]))
            # self.context.set_input_shape(self.engine.get_tensor_name(i), tuple(outputs[i].shape))
            # print(outputs[i].nbytes)

        for i in range(nInput,nInput+nOutput): # TODO 检查输出张量是否与绑定的形状匹配，以确定是否需要重新分配输出缓冲区。
            trt_output_shape = self.context.get_tensor_shape(self.engine.get_tensor_name(i))
            output_idx = i - len(inputs)
            if not (list(trt_output_shape) == list(outputs[output_idx].shape)):
                self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
                self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))
                self.logger.log(trt.Logger.ERROR, "base_output.shape = " + str(outputs[output_idx].shape))
                assert(0)

        # warm up
        self.context.execute_v2(bufferD) # TODO 加载资源，初始化相关资源 执行推理

        T1 = time.perf_counter() # TODO 记录当前时间

        self.context.execute_v2(bufferD) # TODO 执行推理。这次推理的执行时间将被记录下来，用来评估实际的推理速度

        T2 =time.perf_counter() # TODO 记录结束时间

        print("time=" + str((T2-T1) * 1000) + "ms") # TODO 计算执行时间

        for i in range(nInput, nInput + nOutput): # TODO 将输出结果从GPU拷贝到CPU中
            # cuda.memcpy_dtoh(outputs[i - nInput].ravel(), bufferD[i])
            cuda.from_device_like(bufferD[i], outputs[i-nInput])

        for i in range(0, len(outputs)): # TODO 打印输出张量形状和综合
            print("outputs.shape:" + str(outputs[i].shape))
            print("outputs.sum:" + str(outputs[i].sum()))
            # print(outputs[i])

            # print("trt_output.shape:" + str(trt_output.shape))
            # print("trt_output.sum:" + str(trt_output.sum()))
            # print(trt_output.view(-1)[0:10])
            # print("torch.allclose result:" + str(torch.allclose(base_output, trt_output, 1e-05, 1e-03)))
            # print("====================")
        return outputs  # TODO 返回推理结果
        # return torch.allclose(base_output, trt_output, 1e-05, 1e-03)
