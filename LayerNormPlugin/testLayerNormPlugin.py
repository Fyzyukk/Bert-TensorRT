#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import ctypes
import numpy as np
# from cuda import cudart  # TODO 使用 cuda runtime API 我这个pyrotch没有对应cuda版本 所以使用pycuda
import  pycuda.driver as cuda # TODO 两个库中的一些函数不同，例如申请空间，拷贝，释放操作在相应的库中不同的写法
import pycuda.autoinit
import tensorrt as trt

soFilePath      = 'D:/Code/TensorRT-main/LayerNormPlugin/cmake-build-debug/LayerNormPlugin.dll'
nBS             = 4
nSL             = 64
nEmbedding      = 256
epsilon         = 6e-6

np.random.seed(97)

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def layerNormCPU(bufferH):
    _x = bufferH[0]
    nEmbed = bufferH[0].shape[2]
    _0  = np.mean(_x,2)[:,:,np.newaxis]
    _1  = _x - _0
    _2  = _1 * _1
    _3  = np.mean(_2,2)[:,:,np.newaxis]
    _4  = np.array(epsilon,dtype=np.float32)
    _5  = _4.reshape(1,1,1)
    _6  = _3 + _5
    _7  = np.sqrt(_6)
    _8  = 1 / _7                # 1/sqrt(...)
    _9  = _1 * _8
    return _9

def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        # print(c.name)
        if c.name == 'LayerNorm':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')###TODO  初始化plugin
    ctypes.cdll.LoadLibrary(soFilePath)

    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
    config.flags    = 0

    inputTensorList = []
    inputTensorList.append(network.add_input('inputT', trt.float32, [-1,-1,256]) )

    profile = builder.create_optimization_profile()
    profile.set_shape('inputT',[1,4,256],[4,64,256],[16,256,256])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())


    network.mark_output(pluginLayer.get_output(0)) #定义输出张量

    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    inuput_name = engine.get_tensor_name(0)
    context.set_input_shape(inuput_name,[nBS,nSL,nEmbedding])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    
    # nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nInput = 1
    nOutput = 1
    # for i in range(engine.num_bindings):
    #     print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))
    print("input ->" if engine.get_tensor_mode(engine.get_tensor_name(0)) else "output->",engine.get_tensor_dtype(engine.get_tensor_name(0)),engine.get_tensor_shape(engine.get_tensor_name(0)),context.get_tensor_shape(engine.get_tensor_name(0)))

    bufferH = []
    bufferH.append( np.random.rand(nBS,nSL,nEmbedding).astype(np.float32).reshape(nBS,nSL,nEmbedding) * 2 - 1)
    # bufferH.append(np.empty(context.get_tensor_shape(1),dtype=trt.nptype(engine.get_tensor_dtype(1))))
    bufferH.append(np.empty(context.get_tensor_shape(engine.get_tensor_name(1)),dtype=trt.nptype(engine.get_tensor_dtype(engine.get_tensor_name(1)))))

    bufferD = []
    # for i in range(engine.num_bindings):
    #     bufferD.append(cuda.mem_alloc_like(bufferH[i].nbytes)[1])
    #总共两个数据
    for i in range(len(bufferH)):
        bufferD.append(cuda.mem_alloc_like(bufferH[i]))

    # for i in range(nInput):
    #     cuda.to_device(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    for i in range(nInput):
        bufferD[i] = cuda.to_device(bufferH[i])

    #执行 context.execute_v2(bufferD) 时，TensorRT 引擎会使用这些输入数据进行推理，并将结果存储在相应的输出缓冲区中
    #这个方法会启动实际的推理过程，并返回结果到输出缓冲区
    # print("Input data (GPU):", bufferH[0])  # 打印输入数据
    context.execute_v2(bufferD)
    # print("GPU output (after execution):", bufferH[-1])

    # for i in range(nInput, nInput + nOutput):
    #   cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nInput, nInput + nOutput):
        cuda.from_device_like(bufferD[i], bufferH[i])



    print("check result:")
    temp1 = bufferH[-1]
    temp2 = layerNormCPU(bufferH[:1])
    # print("CPU output:", temp2)
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    
    # for b in bufferD:
    #     cuda.cudaFree(b)

if __name__ == '__main__':
    # os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()