学习路线：
--------------------------------------------------------------------
1.了解cuda的基础知识，资料在CUDA-AIMaker2030PDF，密码为AIMaker2030PDF，某蓝的ppt，感觉做的一般，但是聊胜于无。

2.了解tensorRT的基础知识，资料也在上面那个文件夹中

3.了解三种构建trt模型的方式：1）onnx-parser自动构建，2）onnx-surgeon+onnx-parser手动修改构建，3）plugin+API网络层构建。第2，3中都可以需要自己书写plugin，这里就是上面1的cuda基础知识中的kernel。

4.了解模型推理的流程。

部署通用的知识点，为什么要转换成trt，转换成trt过程中做了什么以及用了trt为什么会变快。

---------------------------------------------------------------------


文件观看顺序：
---------------------------------------------------------------------

1.model_to_onnx，训练框架模型转换成onnx模型的步骤和用法。

2.onnx_to_trt，onnx模型转换为trt模型，这个文件中的方法是第一种parser方式。了解模型转换构建的流程：engine,build,config............

3.LayerNormPlugin文件中的书写layernorm的kernel，.h文件是创建的流程，.cu是具体的实现细节。LN_zip为原始版本。知道怎么将这个plugin编译生成.dll(windows)文件或.so(linux)，插入到网络模型中。

4.trt.helper，包含一些常用的网络层以及Infer推理的流程。

5.builder是上面第三种构建的方式，自己搭建网络层。明白网络的每一层是怎么搭建的。

6.calibrator是int8的量化，这部分没调。代码是某蓝的，有很多是错的，调通了，但是结果是推理结果是0。也可能是初学菜鸡的原因。

-----------------------------------------------------------------------

使用步骤：
----------------------------------------------------------------------

1.hugging face 下载Bert模型的pytorch预训练权重文件，放入到bert_pretrain_pytorch文件夹下，里面已经包含了json配置文件和vocab。

2.model存放的是onnx模型和序列化的plan模型

3.onnx-surgeon文件夹存放的是使用onnx-surgeon将econder模型部分二点layernorm节点取出，手动替换为书写的plugin，并在parser过程中自动插入到其中。
这部分的模型是nvidia比赛2022的初赛econder，下面会给出链接，里面还有很多优秀的项目，大家可以去自己学习。

---------------------------------------------------------------------

环境配置：

--------------------------------------------------------------------

1.下载cuda，cudnn，tensorRT这个网上都有教程，一搜就能搜到

2.软件使用visual studio/vscode/clion都可以，本人使用的是clion，因为这个项目给的代码是python的方式，使用clion可以添加python解释器。

3.安装好环境就可以运行项目代码了。

注：安装的tensorRT版本不同，代码里的一些书写形式会有不同，建议去tensorRT的官方手册查看不同版本的实现，修改成对应的形式。某蓝给的也是不正确的，自己按照官方文档修改的。

-------------------------------------------------------------------

nvidia比赛链接：

https://github.com/NVIDIA/trt-samples-for-hackathon-cn。

基础知识就到这里了，下一步会具体做一个vllm和yolo的模型部署项目，做完就去找实习了。学的不是很精，只是了解一些基础。



# BERT Inference Using TensorRT Python API
### 一. 文件信息
1. model2onnx.py 使用pytorch 运行 bert 模型，生成demo 输入输出和onnx模型
2. onnx2trt.py 将onnx，使用onnx-parser转成trt模型，并infer
3. builder.py 输入onnx模型，并进行转换
4. trt_helper.py 对trt的api进行封装，方便调用
5. calibrator.py int8 calibrator 代码
6. 基础款LayerNormPlugin.zip 用于学习的layer_norm_plugin

## 二. 模型信息
### 2.1 介绍
1. 标准BERT 模型，12 层, hidden_size = 768
2. 不考虑tokensizer部分，输入是ids，输出是score
3. 为了更好的理解，降低作业难度，将mask逻辑去除，只支持batch=1 的输入

BERT模型可以实现多种NLP任务，作业选用了fill-mask任务的模型

```
输入：
The capital of France, [mask], contains the Eiffel Tower.

topk10输出：
The capital of France, paris, contains the Eiffel Tower.
The capital of France, lyon, contains the Eiffel Tower.
The capital of France,, contains the Eiffel Tower.
The capital of France, tolilleulouse, contains the Eiffel Tower.
The capital of France, marseille, contains the Eiffel Tower.
The capital of France, orleans, contains the Eiffel Tower.
The capital of France, strasbourg, contains the Eiffel Tower.
The capital of France, nice, contains the Eiffel Tower.
The capital of France, cannes, contains the Eiffel Tower.
The capital of France, versailles, contains the Eiffel Tower.
```

### 2.2 输入输出信息
输入
1. input_ids[1, -1]： int 类型，input ids，从BertTokenizer获得
2. token_type_ids[1, -1]：int 类型，全0
3. position_ids[1, -1]：int 类型，[0, 1, ..., len(input_ids) - 1]

输出
1. logit[1, -1, 768]
 
## 三. 作业内容

### 第6章节作业

### 3.1 学习使用 trt python api 搭建网络
填充trt_helper.py 中的空白函数（addLinear， addSoftmax等）。学习使用api 搭建网络的过程。

### 3.2 编写plugin
trt不支持layer_norm算子，编写layer_norm plugin，并将算子添加到网络中，进行验证。
1. 及格：将 “基础款LayerNormPlugin.zip”中实现的基础版 layer_norm算子 插入到 trt_helper.py addLayerNorm函数中。
2. 优秀：将整个layer_norm算子实现到一个kernel中，并插入到 trt_helper.py addLayerNorm函数中。可以使用testLayerNormPlugin.py对合并后的plugin进行单元测试验证。
3. 进阶：在2的基础上进一步优化，线索见 https://www.bilibili.com/video/BV1i3411G7vN

### 3.3 观察GELU算子的优化过程
1. GELU算子使用一堆**基础算子堆叠实现的**（详细见trt_helper.py addGELU函数），直观上感觉很分散，计算量比较大。  
2. 但在实际build过程中，这些算子会被合并成一个算子。build 过程中需要设置log为trt.Logger.VERBOSE，观察build过程。
3. 体会trt在转换过程中的加速优化操作。

### 第7章节作业

### 3.4 进行 fp16 加速并测试速度
及格标准：设置build_config，对模型进行fp16优化；
优秀标准：编写fp16 版本的layer_norm算子，使模型最后运行fp16版本的layer_norm算子。

### 3.5 进行 int8 加速并测试速度
1. 完善calibrator.py内的todo函数，使用calibrator_data.txt 校准集，对模型进行int8量化加速。

## 四. 深度思考
### 4.1 还有那些算子能合并？
1. emb_layernorm 模块，3个embedding和一个layer_norm，是否可以合并到一个kernel中？
2. self_attention_lay[er 中，softmax和scale操作，是否可以合并到一个kernel中？
3. self_attention_layer，要对qkv进行三次矩阵]()乘和3次转置。三个矩阵乘是否可以合并到一起，相应三个转置是否可以？如果合并了，那么后面的q*k和attn*v，该怎么计算？
4. self_output_layer中，add 和 layer_norm 层是否可以合并？

以上问题的答案，见 https://github.com/NVIDIA/TensorRT/tree/release/6.0/demo/BERT

### 4.2 除了上面那些，还能做那些优化？
1. 增加mask逻辑，多batch一起inference，padding的部分会冗余计算。比如一句话10个字，第二句100个字，那么padding后的大小是[2, 100]，会有90个字的冗余计算。这部分该怎么优化？除了模型内部优化，是否还可以在外部拼帧逻辑进行优化？
2. self_attention_layer层，合并到一起后的QkvToContext算子，是否可以支持fp16计算？int8呢？

以上问题答案，见 https://github.com/NVIDIA/TensorRT/tree/release/8.2/demo/BERT

