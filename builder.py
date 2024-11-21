import argparse
import ctypes
import json
import numpy as np
import os
import os.path
import re
import sys
import time
import onnx
import pycuda.autoinit

# TensorRT
import tensorrt as trt
from tensorrt import ITensor


from calibrator import BertCalibrator as BertCalibrator
from trt_helper import *

import torch
from torch.nn import functional as F
from transformers import BertTokenizer, BertForMaskedLM

# import onnxruntime as ort
import transformers

"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) #TensorRT 日志对象
# TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# handle = ctypes.CDLL("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64/nvinfer_plugin_10.dll", mode=ctypes.RTLD_GLOBAL)
# #通过 ctypes.CDLL 加载 TensorRT 插件库 nvinfer_plugin_10.dll。它允许你加载自定义插件用于扩展 TensorRT
# if not handle:
#     raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.dll` on your LD_LIBRARY_PATH?")
#
# handle = ctypes.CDLL("D:/Code/TensorRT-main/LayerNormPlugin/cmake-build-debug/LayerNormPlugin.dll", mode=ctypes.RTLD_GLOBAL)
# if not handle:
#     raise RuntimeError("Could not load plugin library. Is `LayerNorm.so` on your LD_LIBRARY_PATH?")
# TODO TensorRT 插件系统，并获取插件注册表。插件注册表允许你获取和使用自定义插件
trt.init_libnvinfer_plugins(TRT_LOGGER, "")  ##TODO  初始化plugin
plg_registry = trt.get_plugin_registry()

class BertConfig: # TODO BertConfig 类用于存储 Bert 模型的配置。通过加载一个 JSON 文件 bert_config_path 读取模型配置，并初始化模型的参数
    def __init__(self, bert_config_path, use_fp16, use_int8, use_strict):
        with open(bert_config_path, "r") as f:
            data = json.load(f)
            self.num_attention_heads = data["num_attention_heads"]
            self.hidden_size = data["hidden_size"]
            self.intermediate_size = data["intermediate_size"]
            self.num_hidden_layers = data["num_hidden_layers"]
            self.head_size = self.hidden_size // self.num_attention_heads
            self.use_fp16 = use_fp16
            self.use_int8 = use_int8
            self.use_strict = use_strict
            self.is_calib_mode = False
def set_tensor_name(tensor:ITensor, prefix, name): # TODO set_tensor_name会为给定的张量添加前缀和名称，通过为张量的名称添加指定的前缀和名称，生成一个带有前缀的唯一标识
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0) -> ITensor : # TODO 专门用来为网络层的输出张量设置名称的。
    set_tensor_name(layer.get_output(out_idx), prefix, name)       # TODO 它接收一个网络层 layer，并通过调用 get_output(out_idx) 获取指定索引的输出张量，然后调用 set_tensor_name 为该输出张量设置名称。这个函数的目的是方便为特定层的输出张量设置名称

def set_output_range(layer, maxval, out_idx = 0): # TODO 设定层的输出范围
    layer.get_output(out_idx).set_dynamic_range(-maxval, maxval)

def get_mha_dtype(config): # TODO 根据配置设置数据类型（dtype）用于多头自注意力（Multi-Head Attention，MHA）层，并返回相应的数值类型
    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16
    # Multi-head attention doesn't use INT8 inputs and output by default unless it is specified.
    if config.use_int8 and config.use_int8_multihead and not config.is_calib_mode:
        dtype = trt.int8
    return int(dtype)

def custom_fc(config, network, input_tensor, out_dims, W): # TODO 在 TensorRT 中创建一个自定义的插件层（fcplugin），并将其添加到一个神经网络中
    pf_out_dims = trt.PluginField("out_dims", np.array([out_dims], dtype=np.int32), trt.PluginFieldType.INT32)
    pf_W = trt.PluginField("W", W.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([1 if config.use_fp16 else 0], np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_out_dims, pf_W, pf_type])
    # fc_plugin = fc_plg_creator.create_plugin("fcplugin", pfc)
    plugin_registry = trt.get_plugin_registry()
    plugin_creator = plugin_registry.get_plugin_creator()
    fc_plugin = plugin_creator.create_plugin('fcplugin',pfc)
    plug_inputs = [input_tensor]
    out_dense = network.add_plugin_v2(plug_inputs, fc_plugin)
    return out_dense # TODO 返回一个层trt.ILayer 对象，表示一个自定义的插件层。

def self_attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask): # TODO 函数实现了自注意力机制的计算，包括计算Query（Q）、Key（K）、Value（V）并进行矩阵乘法
    # TODO network_helper：这是一个帮助函数或类，封装了一些常用的 TensorRT 操作（如矩阵乘法、线性层、转换操作等）。
    # TODO prefix：该参数是用来为每个操作层命名的前缀。通常这个前缀用于从权重字典中获取特定的权重和偏置。
    # TODO config：包含模型配置的字典或对象，包括注意力头的数量 (num_attention_heads) 和每个头的大小 (head_size) 等。
    # TODO weights_dict：包含所有网络权重的字典，按名称存储权重（例如，查询、键、值的权重矩阵和偏置）。
    # TODO input_tensor：输入的张量，通常是前一层的输出。
    # TODO imask：注意力掩码，通常用于遮蔽掉某些位置（如填充位置）
    num_heads = config.num_attention_heads
    head_size = config.head_size

    q_w = weights_dict[prefix + "attention_self_query_kernel"]
    q_b = weights_dict[prefix + "attention_self_query_bias"]
    # TODO q_w 和 q_b：从 weights_dict 获取查询（Query）权重矩阵和偏置。prefix 用来为这些权重加上唯一的前缀（例如 "layer1_attention_self_query_kernel"
    q = network_helper.addLinear(input_tensor, q_w, q_b)
    q = network_helper.addShuffle(q, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_q_view_transpose")

    k_w = weights_dict[prefix + "attention_self_key_kernel"]
    k_b = weights_dict[prefix + "attention_self_key_bias"]
    # TODO 计算查询类似，k_w 和 k_b 获取键（Key）权重和偏置
    k = network_helper.addLinear(input_tensor, k_w, k_b)
    k = network_helper.addShuffle(k, None, (0, -1, num_heads, head_size), (0, 2, 3, 1), "att_k_view_and transpose")
    # k = network_helper.addShuffle(k, None, (0, -1, self.h, self.d_k), (0, 2, 3, 1), "att_k_view_and transpose")

    v_w = weights_dict[prefix + "attention_self_value_kernel"]
    v_b = weights_dict[prefix + "attention_self_value_bias"]
    # TODO v_w 和 v_b 获取值（Value）的权重和偏置
    v = network_helper.addLinear(input_tensor, v_w, v_b)
    v = network_helper.addShuffle(v, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_v_view_and transpose")

    scores = network_helper.addMatMul(q, k, "q_mul_k") # TODO 计算注意力得分（Q * K）

    scores = network_helper.addScale(scores, 1/math.sqrt(head_size)) # TODO 缩放得分

    attn = network_helper.addSoftmax(scores, dim=-1) # TODO 应用 Softmax

    attn = network_helper.addMatMul(attn, v, "matmul(p_attn, value)") # TODO 计算最终输出（attn * V）

    attn = network_helper.addShuffle(attn, (0, 2, 1, 3), (0, -1, num_heads * head_size), None, "attn_transpose_and_reshape") # TODO 调整输出张量的形状

    return attn # TODO 返回trt.ITensor 类型的注意力输出张量

def self_output_layer(network_helper, prefix, config, weights_dict, hidden_states, input_tensor): # TODO 实现了BERT中的自注意力机制（Self-Attention）输出的处理
    # TODO hidden_states：自注意力层的输出，包含了当前层的隐藏状态。 这个就是上面self_attention_layer返回的。
    # TODO input_tensor：来自上一层（如输入序列或前一层）的张量，通常是对当前输入的残差连接。和上面self_attention_layer的输入相同。

    out_w = weights_dict[prefix + "attention_output_dense_kernel"]
    out_b = weights_dict[prefix + "attention_output_dense_bias"]
    # TODO 从 weights_dict 中获取输出层的权重（attention_output_dense_kernel）和偏置（attention_output_dense_bias）。这些是用于执行线性变换的权重和偏置
    out = network_helper.addLinear(hidden_states, out_w, out_b) # TODO 计算 Attention 输出层的线性变换

    out = network_helper.addAdd(out, input_tensor) # TODO 加入残差连接

    gamma = weights_dict[prefix + "attention_output_layernorm_gamma"]
    beta = weights_dict[prefix + "attention_output_layernorm_beta"]
    out = network_helper.addLayerNorm(out, gamma, beta) # TODO 应用 LayerNorm（层归一化）

    return out # TODO 返回trt.ITensor 类型的输出张量

def attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask):# TODO 实现了BERT中的注意力层，由两部分组成：
    attn = self_attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask) # TODO 计算自注意力输出（Self-Attention）

    out = self_output_layer(network_helper, prefix, config, weights_dict, attn, input_tensor) # TODO 计算输出层（Output Layer）

    return out # TODO 返回trt.ITensor 类型的输出张量


def transformer_layer(network_helper, prefix, config, weights_dict, input_tensor, imask):
    # TODO 实现了BERT中的Transformer层，包括以下步骤
    # TODO 注意力层：首先调用attention_layer计算注意力输出（包括自注意力和输出层的处理）。
    # TODO 中间层（BertIntermediate）：对注意力层的输出应用线性变换，得到中间结果。这里使用了intermediate_dense_kernel和intermediate_dense_bias。
    # TODO 激活函数：对中间结果应用GELU激活函数。
    # TODO 输出层（BertOutput）：将中间层的输出通过线性变换得到最终的输出。使用output_dense_kernel和output_dense_bias作为权重和偏置。
    # TODO 残差连接和层归一化：将输出层的结果与注意力层的输出进行加法（残差连接），并进行层归一化
    num_heads = config.num_attention_heads
    head_size = config.head_size

    attention_output = attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask) # TODO 注意力层（Attention Layer）

    #  TODO BertIntermediate（中间层）
    intermediate_w = weights_dict[prefix + "intermediate_dense_kernel"]
    intermediate_w = np.transpose(intermediate_w)
    # TODO np.transpose() 转置权重，以确保权重的形状与输入张量匹配
    intermediate_b = weights_dict[prefix + "intermediate_dense_bias"]
    # TODO intermediate_dense_kernel 和 intermediate_dense_bias 是中间层的线性变换权重和偏置
    intermediate_output = network_helper.addLinear(attention_output, intermediate_w, intermediate_b)
    # TODO  intermediate_output = attention_output * intermediate_dense_kernel + intermediate_dense_bias
    intermediate_output = network_helper.addGELU(intermediate_output) # TODO GELU 激活函数

    #  TODO BertOutput（输出层）
    output_w = weights_dict[prefix + "output_dense_kernel"]
    output_w = np.transpose(output_w)
    output_b = weights_dict[prefix + "output_dense_bias"]
    layer_output = network_helper.addLinear(intermediate_output, output_w, output_b)

    layer_output = network_helper.addAdd(layer_output, attention_output) # TODO 残差连接（Residual Connection）
    # TODO layer_out = layer_out + attention_output

    gamma = weights_dict[prefix + "output_layernorm_gamma"]
    beta = weights_dict[prefix + "output_layernorm_beta"]
    layer_output = network_helper.addLayerNorm(layer_output, gamma, beta) # TODO 层归一化（Layer Normalization）

    return layer_output # TODO 返回trt.ITensor 类型的输出张量

def transformer_output_layer(network_helper, config, weights_dict, input_tensor):
    # TODO 该函数实现了Transformer最后的输出层，用于计算模型的最终预测（如分类任务中的预测类别）
    # TODO 中间层：对输入进行线性变换，使用cls_predictions_transform_dense_kernel和cls_predictions_transform_dense_bias作为权重和偏置。
    # TODO 激活函数：应用GELU激活函数，增加非线性。
    # TODO 层归一化：对结果进行层归一化，使用cls_predictions_transform_layernorm_gamma和cls_predictions_transform_layernorm_beta作为参数。
    # TODO 最终输出层：将经过层归一化的输出与embeddings_word_embeddings进行线性变换，生成最终输出。这里用cls_predictions_bias作为偏置
    num_heads = config.num_attention_heads
    head_size = config.head_size

    #  TODO BertIntermediate（中间层）
    dense_w = weights_dict["cls_predictions_transform_dense_kernel"]
    dense_w = np.transpose(dense_w)
    dense_b = weights_dict["cls_predictions_transform_dense_bias"]
    dense_output = network_helper.addLinear(input_tensor, dense_w, dense_b)

    dense_output = network_helper.addGELU(dense_output) # TODO 激活函数（GELU）

    gamma = weights_dict["cls_predictions_transform_layernorm_gamma"]
    beta = weights_dict["cls_predictions_transform_layernorm_beta"]
    layer_output = network_helper.addLayerNorm(dense_output, gamma, beta) # TODO 层归一化（Layer Normalization）

    #  TODO BertOutput（输出层）
    output_w = weights_dict["embeddings_word_embeddings"]
    output_w = np.transpose(output_w)
    output_b = weights_dict["cls_predictions_bias"]
    layer_output = network_helper.addLinear(layer_output, output_w, output_b) #  TODO 最终输出层（Output Layer）

    return layer_output # TODO 返回trt.ITensor 类型的输出张量

def bert_model(network_helper, config, weights_dict, input_tensor, input_mask):
    # TODO input_tensor：模型的输入张量，通常是词嵌入（Embedding）后的张量，形状为 (batch_size, seq_len, embedding_dim)。
    # TODO input_mask：是用于标识哪些词是实际输入（非 padding）的掩码张量，形状通常为 (batch_size, seq_len)，在计算时用来屏蔽掉 padding 部分的计算
    # TODO 构建了整个BERT模型的结构。它通过循环迭代模型的隐藏层数（num_hidden_layers），逐层调用transformer_layer函数构建每一层的结构
    """
    Create the bert model
    """
    prev_input = input_tensor # TODO 存储上一层的输出。初始化时，上一层的输出就是输入张量 input_tensor
    for layer in range(0, config.num_hidden_layers):
        ss = "l{}_".format(layer) # TODO 每一层生成一个唯一的前缀（例如：l0_、l1_），确保每一层的权重、张量名称等唯一
        prev_input = transformer_layer(network_helper, ss, config,  weights_dict, prev_input, input_mask)
        # TODO 每一层的输入是上一层的输出，第一层的输入是 input_tensor，后续每一层的输入都是上一层的输出。通过 prev_input = transformer_layer(...) 将输入传入

    return prev_input # TODO 返回trt.ITensor 类型的输出张量 形状为 (batch_size, seq_len, hidden_size) 的张量。它包含了每个输入 token（词）的上下文相关的表示


def onnx_to_trt_name(onnx_name):# TODO 模型从 ONNX 权重加载和处理到 TensorRT 构建过程中的一些关键操作
    """
    Converting variables in the onnx checkpoint to names corresponding to the naming convention used in the TF version, expected by the builder
    """
    onnx_name = onnx_name.lower()
    toks = [t.strip('_') for t in onnx_name.split('.')]
    # TODO 通过 . 分割 onnx_name 字符串，将其拆分为多个部分，并去掉每个部分前后的下划线（_）。最终的结果是一个列表 toks，每个元素都是 ONNX 权重名称的一个组成部分

    if toks[0] == 'bert':
        if toks[1] == 'embeddings': #embedding
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            else: #embeddings: drop "_weight" suffix
                toks = toks[:-1]
            toks = toks[1:]
        elif toks[1] == 'encoder': #transformer
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            elif (toks[-2] == 'dense' or toks[-2] in {'key', 'value', 'query'}) and toks[-1] == 'weight':
                toks[-1] = 'kernel'
            elif (toks[-3] == 'dense' or toks[-3] in {'key', 'value', 'query'}) and toks[-1] == 'amax':
                if toks[-2] == 'weight_quantizer':
                    toks[-2] = 'kernel'
                elif toks[-2] == 'input_quantizer':
                    toks[-2] = 'input'
            toks = toks[3:]
            toks[0] = 'l{}'.format(int(toks[0]))
    elif 'cls' in onnx_name:
        if 'transform' in onnx_name:
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            elif (toks[-2] == 'dense' and toks[-1] == 'weight'):
                toks[-1] = 'kernel'
        # else:
            # name = 'pooler_bias' if toks[-1] == 'bias' else 'pooler_kernel'
    else:
        print("Encountered unknown case:", onnx_name)
        assert(False)
    parsed = '_'.join(toks)
    return parsed # TODO 返回处理后的权重名称 实际上就是转换成上面构建网络所需要的weights_dict

def load_onnx_weights_and_quant(path, config):# TODO 加载ONNX模型中的权重，并根据上一步的命名规则将权重存储到字典中
    # TODO path：表示 ONNX 模型文件的路径。
    # TODO config：配置对象，包含模型的各种超参数和设置，例如 num_attention_heads、head_size、hidden_size 等
    """
    Load the weights from the onnx checkpoint
    """
    N = config.num_attention_heads
    H = config.head_size
    hidden_size = config.hidden_size

    model = onnx.load(path)
    weights = model.graph.initializer # TODO 取模型的 初始化器，即模型中的权重。这些权重存储在 initializer 中

    tensor_dict = {}
    for w in weights:
        if "position_ids" in w.name:
            continue

        a = onnx_to_trt_name(w.name) # TODO 调用之前定义的 onnx_to_trt_name 函数，将 ONNX 权重名称 w.name 转换为 TensorRT 需要的名称格式
        # print(w.name + " " + str(w.dims))
        print(a + " " + str(w.dims)) # TODO 打印转换后的权重名称 a 和该权重的形状 w.dims，以便进行调试和检查
        b = np.frombuffer(w.raw_data, np.float32).reshape(w.dims)
        # TODO w.raw_data 是存储权重数据的原始二进制数据（通常是一个序列化的 NumPy 数组），通过 np.frombuffer 将其读取为 np.float32 类型的数组
        # TODO 使用 reshape(w.dims) 将其按照权重的维度重新排列，确保它的形状与 w.dims 中的指定形状一致
        tensor_dict[a] = b # TODO 将转换后的权重数据 b 存储到字典 tensor_dict 中，键为转换后的权重名称 a

    weights_dict = tensor_dict # TODO tensor_dict 赋值给 weights_dict，作为最终返回的结果

    TRT_LOGGER.log(TRT_LOGGER.INFO, "Found {:} entries in weight map".format(len(weights_dict)))
    return weights_dict # TODO 返回权重字典

def emb_layernorm(network_helper, config, weights_dict, builder_config, sequence_lengths, batch_sizes):
    # 实现了BERT模型中的词嵌入层和层归一化
    # int8 only support some of the sequence length, we dynamic on sequence length is not allowed.
    # TODO 创建输入节点 这些输入节点的形状都是 (1, -1)，表示批大小为 1，序列长度可变
    input_ids = network_helper.addInput(name="input_ids", dtype=trt.int32, shape=(1, -1))
    token_type_ids = network_helper.addInput(name="token_type_ids", dtype=trt.int32, shape=(1, -1))
    position_ids = network_helper.addInput(name="position_ids", dtype=trt.int32, shape=(1, -1))

    # TODO 获取了 BERT 模型的 词嵌入矩阵、位置嵌入矩阵 和 token 类型嵌入矩阵
    word_embeddings = weights_dict["embeddings_word_embeddings"]
    position_embeddings = weights_dict["embeddings_position_embeddings"]
    token_type_embeddings = weights_dict["embeddings_token_type_embeddings"]
    print(word_embeddings)

    # TODO 输入的 token IDs、token 类型 IDs 和 位置 IDs 映射到它们对应的嵌入向量, 操作本质上是对输入的 ID 进行 查表，查找并返回对应的嵌入向量
    input_embeds = network_helper.addEmbedding(input_ids, word_embeddings)
    token_type_embeds = network_helper.addEmbedding(token_type_ids, token_type_embeddings)
    position_embeds = network_helper.addEmbedding(position_ids, position_embeddings)

    embeddings = network_helper.addAdd(input_embeds, position_embeds) # 合并嵌入向量
    embeddings = network_helper.addAdd(embeddings, token_type_embeds)

    gamma = weights_dict["embeddings_layernorm_gamma"]
    beta = weights_dict["embeddings_layernorm_beta"]
    out = network_helper.addLayerNorm(embeddings, gamma, beta)

    return out # TODO 返回是trt.ITensor

def build_engine(workspace_size, config, weights_dict, vocab_file, calibrationCacheFile, calib_num):
    # TODO 该函数用来构建TensorRT的推理引擎。它配置了FP16/INT8的精度选
    # TODO workspace_size: 需要为 TensorRT 推理引擎分配的工作空间大小（以字节为单位）。
    # TODO config: 一个包含模型配置的对象，决定了模型构建时的设置（如是否使用 FP16、INT8、是否启用严格模式等）。
    # TODO weights_dict: 存储模型权重的字典。
    # TODO vocab_file: 词汇文件，通常用于加载 BERT 的词汇表。
    # TODO calibrationCacheFile: 用于 INT8 量化的缓存文件，存储量化过程中计算的标定信息。
    # TODO calib_num: 在进行 INT8 量化时使用的样本数量

    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    max_seq_length = 200
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,workspace_size * (1024 * 1024))
        if config.use_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        if config.use_int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)

            calibrator = BertCalibrator("calibrator_data.txt", "bert-base-uncased", calibrationCacheFile, 1, max_seq_length, 1000)
            builder_config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
            builder_config.int8_calibrator = calibrator

        if config.use_strict:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        # builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # only use the largest sequence when in calibration mode
        if config.is_calib_mode:
            sequence_lengths = sequence_lengths[-1:]

        # TODO network_helper 是一个工具类或辅助对象，用于简化和管理 TensorRT 网络操作，比如添加层、设置输入和输出等
        network_helper = TrtNetworkHelper(network, plg_registry, TRT_LOGGER)

        # TODO 构建 BERT 模型：
        # 创建 BERT 模型的 词嵌入层 和 层归一化
        embeddings = emb_layernorm(network_helper, config, weights_dict, builder_config, None, None)

        # 调用 bert_model 函数，构建整个 BERT 网络（包括 Transformer 层）
        bert_out = bert_model(network_helper, config, weights_dict, embeddings, None)
        # network_helper.markOutput(bert_out)

        # 使用 transformer_output_layer 获取模型的输出层（用于分类等任务）
        cls_output = transformer_output_layer(network_helper, config, weights_dict, bert_out)

        # TODO 设置网络的输出：
        network_helper.markOutput(cls_output)

        profile = builder.create_optimization_profile()
        min_shape = (1, 1)
        opt_shape = (1, 50)
        max_shape = (1, max_seq_length)
        profile.set_shape("input_ids", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("position_ids", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("token_type_ids", min=min_shape, opt=opt_shape, max=max_shape)
        builder_config.add_optimization_profile(profile)

        build_start_time = time.time()
        # TODO 构建引擎
        engine = builder.build_engine_with_config(network, builder_config)
        build_time_elapsed = (time.time() - build_start_time)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))
        if config.use_int8:
            calibrator.free()
        return engine

def generate_calibration_cache(sequence_lengths, workspace_size, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num):
    # TODO 生成用于INT8量化的校准缓存。这个缓存是在FP32模式下生成的，但最终的推理引擎会使用INT8精度
    """
    BERT demo needs a separate engine building path to generate calibration cache.
    This is because we need to configure SLN and MHA plugins in FP32 mode when
    generating calibration cache, and INT8 mode when building the actual engine.
    This cache could be generated by examining certain training data and can be
    reused across different configurations.
    """
    # dynamic shape not working with calibration, so we need generate a calibration cache first using fulldims network
    if not config.use_int8 or os.path.exists(calibrationCacheFile):
        return calibrationCacheFile

    # generate calibration cache
    saved_use_fp16 = config.use_fp16
    config.use_fp16 = False
    config.is_calib_mode = True

    with build_engine([1], workspace_size, sequence_lengths, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num) as engine:
        TRT_LOGGER.log(TRT_LOGGER.INFO, "calibration cache generated in {:}".format(calibrationCacheFile))

    config.use_fp16 = saved_use_fp16
    config.is_calib_mode = False

def test_text(infer_helper, BERT_PATH):
    # TODO 行文本推理测试，验证模型的输出结果
    # TODO infer_helper：一个用于执行推理的帮助类或对象。
    # TODO BERT_PATH：BERT 模型文件的路径，用于加载预训练的模型和分词器
    print("==============model test===================")
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)

    input_ids = encoded_input['input_ids'].int().detach().numpy()
    token_type_ids = encoded_input['token_type_ids'].int().detach().numpy()
    position_ids = torch.arange(0, encoded_input['input_ids'].shape[1]).int().view(1, -1).numpy()
    input_list = [input_ids, token_type_ids, position_ids]

    output = infer_helper.infer(input_list)
    print(output)

    logits = torch.from_numpy(output[0])
    print(logits)

    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)

def test_case_data(infer_helper, case_data_path):
    # TODO 使用预先准备好的测试数据进行推理并验证推理结果
    # TODO infer_helper：用于执行推理的帮助类或对象，通常会封装推理逻辑。
    # TODO case_data_path：测试数据的路径，包含了输入数据和参考输出数据
    print("==============test_case_data===================")
    case_data = np.load(case_data_path)

    input_ids = case_data['input_ids']
    token_type_ids = case_data['token_type_ids']
    position_ids = case_data['position_ids']
    print(input_ids)
    print(input_ids.shape)
    print(token_type_ids)
    print(token_type_ids.shape)
    print(position_ids)
    print(position_ids.shape)

    logits_output = case_data['logits']

    trt_outputs = infer_helper.infer([input_ids, token_type_ids, position_ids])
    # infer_helper.infer([input_ids], [output_start])

    rtol = 1e-02
    atol = 1e-02

    # res = np.allclose(logits_output, trt_outputs[0], rtol, atol)
    # print ("Are the start outputs are equal within the tolerance:\t", res)
    print(logits_output.sum())
    print(logits_output)
    print(trt_outputs[0].sum())
    print(trt_outputs[0])

def main():
    # TODO 解析命令行参数
    parser = argparse.ArgumentParser(description="TensorRT BERT Sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-x", "--onnx", required=False,default='D:/Code/TensorRT-main/model/model.onnx', help="The ONNX model file path.")
    parser.add_argument("-o", "--output", required=False, default="D:/Code/TensorRT-main/model/bert_base.engine", help="The bert engine file, ex bert.engine")
    parser.add_argument("-c", "--config-dir", required=False,default='D:/Code/TensorRT-main/bert_pretrain_pytorch',
                        help="The folder containing the bert_config.json, which can be downloaded e.g. from https://github.com/google-research/bert#pre-trained-models or by running download_models.py in dle/TensorFlow/LanguageModeling/BERT/data/pretrained_models_google")
    parser.add_argument("-f", "--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-i", "--int8", action="store_true", help="Indicates that inference should be run in INT8 precision", required=False)
    parser.add_argument("-t", "--strict", action="store_true", help="Indicates that inference should be run in strict precision mode", required=False)
    parser.add_argument("-w", "--workspace-size", default=1000, help="Workspace size in MiB for building the BERT engine", type=int)
    parser.add_argument("-v", "--vocab-file", default="D:/Code/TensorRT-main/bert_pretrain_pytorch/vocab.txt", help="Path to file containing entire understandable vocab", required=False)
    parser.add_argument("-n", "--calib-num", default=100, help="calibration batch numbers", type=int)
    parser.add_argument("-p", "--calib-path", help="calibration cache path", required=False)

    args, _ = parser.parse_known_args()

    # args.batch_size = args.batch_size or [1]
    # args.sequence_length = args.sequence_length or [128]

    # cc = pycuda.autoinit.device.compute_capability()
    # if cc[0] * 10 + cc[1] < 75 and args.force_int8_multihead:
        # raise RuntimeError("--force-int8-multihead option is only supported on Turing+ GPU.")
    # if cc[0] * 10 + cc[1] < 72 and args.force_int8_skipln:
        # raise RuntimeError("--force-int8-skipln option is only supported on Xavier+ GPU.")

    bert_config_path = os.path.join(args.config_dir, "config.json")
    # 设置 BERT 配置文件路径：根据 config-dir 参数生成配置文件路径，并输出日志信息，显示所使用的配置文件路径
    TRT_LOGGER.log(TRT_LOGGER.INFO, "Using configuration file: {:}".format(bert_config_path))

    # 创建 BERT 配置对象：加载配置文件，并根据用户提供的命令行参数（如是否使用 FP16/INT8）创建 BertConfig 配置对象
    config = BertConfig(bert_config_path, args.fp16, args.int8, args.strict)

    if args.calib_path != None:
        calib_cache = args.calib_path
    else:
        calib_cache = "BertL{}H{}A{}CalibCache".format(config.num_hidden_layers, config.head_size, config.num_attention_heads)

    if args.onnx != None:
        weights_dict = load_onnx_weights_and_quant(args.onnx, config) # TODO 加载 ONNX 权重
    else:
        raise RuntimeError("You need either specify ONNX using option --onnx to build TRT BERT model.")

    with build_engine(args.workspace_size, config, weights_dict, args.vocab_file, calib_cache, args.calib_num) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize() # TODO 序列化引擎
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")

    infer_helper = InferHelper(args.output, TRT_LOGGER)

    test_case_data(infer_helper, args.config_dir + "/case_data.npz")

    test_text(infer_helper, args.config_dir)

if __name__ == "__main__":
    main()
