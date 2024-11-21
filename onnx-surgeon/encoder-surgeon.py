
import onnx
import onnx_graphsurgeon as gs

sourceOnnx = './encoder.onnx'
destinationOnnx = './encoder_v1.onnx'

model = onnx.load(sourceOnnx)
graph = gs.import_onnx(model)

def econder_surgeon_layer_norm(graph):
    start_node = None
    end_node = None

    weight_node = None
    bias_node = None
    layer_norm_idx = 0

    for node in graph.nodes:
        if node.op == 'ReduceMean' and node.o(0).op == 'Sub':
            start_node = node
            sub_node = node.o(0)
            if sub_node.o(0).op == 'Pow':
                pow_node = sub_node.o(0)
                if pow_node.o(0).op == 'ReduceMean':
                    reducemean_node = pow_node.o(0)
                    if reducemean_node.o(0).op == 'Add':
                        add_node = reducemean_node.o(0)
                        if add_node.o(0).op == 'Sqrt':
                            sqrt_node = add_node.o(0)
                            if sqrt_node.o(0).op == 'Div':
                                div_node = sqrt_node.o(0)
                                if div_node.o(0).op == 'Mul':
                                    mul_node = div_node.o(0)
                                    weight_node = mul_node
                                    if mul_node.o(0).op == 'Add':
                                        add_node = mul_node.o(0)
                                        bias_node = add_node
                                        end_node = bias_node

                                        layer_norm_plugin = gs.Node('LayerNorm','LayerNorm-'+ str(layer_norm_idx))
                                        layer_norm_idx += 1
                                        graph.nodes.append(layer_norm_plugin)

                                        print(start_node)
                                        print(end_node)
                                        print(weight_node)
                                        print(bias_node)
                                        print('----------------------------------')
                                        layer_norm_plugin.inputs = [start_node.inputs[0],weight_node.inputs[1],bias_node.inputs[1]]
                                        layer_norm_plugin.outputs = end_node.outputs

                                        start_node.inputs = []
                                        end_node.outputs = []

                                        start_node = None
                                        end_node = None
                                        weight_node = None
                                        bias_node = None

    return graph

if __name__ == '__main__':
    graph = econder_surgeon_layer_norm(graph)
    graph.cleanup()
    onnx.save(gs.export_onnx(graph),destinationOnnx)