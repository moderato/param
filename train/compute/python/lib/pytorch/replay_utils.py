# N.B. Exgr utils required. Integration to Pytorch WIP.
import torch
import io, json, re
from pprint import pprint
from fbgemm_gpu.split_table_batched_embeddings_ops import PoolingMode
from .exec_graph_utils import NodeType
from ...lib.config import make_op_config
from ...lib.pytorch.config_util import (
    create_op_args,
    create_op_info,
    get_benchmark_options,
)


# TODO: Add all torch dtypes to here
TORCH_DTYPES_RNG = {
    "bool": (torch.bool, torch.ones),
    "int8": (torch.int8, torch.ones),
    "half": (torch.half, torch.ones),
    "int": (torch.int, torch.ones),
    "long": (torch.int64, torch.ones),
    "long int": (torch.int64, torch.ones),
    "float": (torch.float32, torch.randn),
    "double": (torch.float64, torch.randn)
}


def is_tensor_list(n, ip):
    return isinstance(ip, int) and 'GenericList[Tensor' in n.input_types[ip]


def is_tensor(n, ip):
    return isinstance(ip, int) and 'Tensor' in n.input_types[ip] and 'GenericList' not in n.input_types[ip]


def is_op(node, strict=False):
    if not strict:
        return node.type == NodeType.OPERATOR
    return node.type == NodeType.OPERATOR and (
                node.parent is not None and \
                node.parent.type != NodeType.OPERATOR\
            )


def has_backward_parent(op):
    if not op.parent or op.parent.id == op.id: # Top op
        return False
    if is_backward_parent(op):
        return True
    return has_backward_parent(op.parent)


def is_backward_parent(op):
    return "autograd::engine::evaluate_function: " in op.name or \
            "Optimizer" in op.name


def is_backward_aten(op):
    return op.name.startswith("aten::") and \
            has_backward_parent(op)


def fbgemm_input_args_indices(n):
    idx_list = None
    if 'sgd' in n.name:
        # exact_sgd: 11: indices, 12: offsets, 14: indice_weights
        if n.inputs[14] == '<None>':
            idx_list = [11, 12]
        else:
            idx_list = [11, 12, 14]
    return idx_list


def is_fbgemm(op):
    return 'fbgemm::split_embedding_codegen_lookup_' in op.name


def is_fbgemm_unweighted(op):
    return is_fbgemm(op) and len(fbgemm_input_args_indices(op)) == 2


# TODO: Hopefully merge is_fbgemm and skip_op
def skip_op(op):
    # Workaround: skip bounds check indices and other ops under embedding lookup module
    return "bounds_check_indices" in op.name or \
            ("fbgemm" not in op.name and op.parent is not None and "embedding_lookup" in op.parent.name)


def is_qualified(op):
    return not skip_op(op) and (is_backward_aten(op) or (is_op(op, strict=True) and not is_backward_parent(op)))


def trace_handler(prof):
    # print(prof.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")


def another_trace_handler(text=""):
    def handle_fn(prof):
        # print(prof.key_averages().table(
        #     sort_by="self_cuda_time_total", row_limit=-1))
        prof.export_chrome_trace("/tmp/test_trace_replay_{}.json".format(text))
    return handle_fn


def execution_graph_handler(output_file_name):
    print(f"pytroch execution graph output: {output_file_name}")
    found_root_node = False
    with io.open(output_file_name, 'r') as f:
        eg_graph = json.load(f)
        assert "nodes" in eg_graph
        nodes = eg_graph["nodes"]
        for n in nodes:
            assert "name" in n
            if "__ROOT_PROCESS__" in n["name"]:
                found_root_node = True
    assert found_root_node


def get_input_tensors(n):
    if is_fbgemm(n):
        idx_list = fbgemm_input_args_indices(n)
        return zip([n.input_types[x] for x in idx_list],
                    [tuple(n.inputs[x]) if isinstance(n.inputs[x], list) else n.inputs[x] for x in idx_list],
                    [n.input_shapes[x] for x in idx_list])
    return n.get_input_tensors()


def get_output_tensors(n):
    if is_fbgemm(n):
        return zip(n.output_types,
                    [tuple(x) for x in n.outputs],
                    n.output_shapes)
    return n.get_output_tensors()


def c10_type_to_str(t):
    if t == "c10:Half":
        return "fp16"
    return "fp32"
    # raise ValueError("c10 type not supported!")


def get_optimizer_from_fbgemm_function_name(s):
    opt = s[39:].split("_")[0] # strip 'fbgemm::split_embedding_codegen_lookup_'
    return "exact_{}".format(opt) # Workaround, should be more accurate


def build_func(n):
    if is_fbgemm(n):
        return build_fbgemm_func(n)
    return build_torchscript_func(n)


def get_fbgemm_info(n):
    rows = [{"value": int(e)} for e in n.parent.inputs[0].split('-')]
    num_tables = len(rows)
    dim = [n.inputs[8]] * num_tables # Assuming all Ds are the same
    batch_size = int((n.input_shapes[12][0] - 1) / num_tables)
    pooling_factor = [{"value": int(n.input_shapes[11][0] / batch_size / num_tables)}] * num_tables
    weighted = "Float" not in n.input_types[1] # e.g. c10:Half
    weights_precision = c10_type_to_str(n.input_types[1])
    optimizer = get_optimizer_from_fbgemm_function_name(n.name)
    return rows, num_tables, dim, batch_size, pooling_factor, weighted, weights_precision, optimizer


def build_fbgemm_func(n):
    assert n.parent is not None
    op_name = "SplitTableBatchedEmbeddingBagsCodegen"
    op_info = create_op_info()
    run_options = get_benchmark_options()
    run_options["device"] = "cuda"
    op_config = make_op_config(op_name, op_info, run_options["device"])

    rows, num_tables, dim, _, _, weighted, weights_precision, optimizer = \
        get_fbgemm_info(n)

    op_config.op.build(
        num_tables,
        [r['value'] for r in rows],
        dim,
        PoolingMode.SUM,
        weighted,
        weights_precision,
        optimizer,
    )

    return op_config.op.forward, len(n.outputs)


def generate_fbgemm_tensors(n):
    assert n.parent is not None
    op_name = "SplitTableBatchedEmbeddingBagsCodegen"
    op_info = create_op_info()
    op_info[
        "input_data_generator"
    ] = "SplitTableBatchedEmbeddingBagsCodegenInputDataGenerator"
    run_options = get_benchmark_options()
    run_options["device"] = "cuda"
    op_config = make_op_config(op_name, op_info, run_options["device"])

    rows, num_tables, dim, batch_size, pooling_factor, weighted, weights_precision, optimizer = \
        get_fbgemm_info(n)

    data_generator_config = create_op_args(
        [
            {"type": "int", "name": "num_tables", "value": num_tables},
            {"type": "int", "name": "rows", "value": rows},
            {"type": "int", "name": "dim", "value": dim},
            {"type": "int", "name": "batch_size", "value": batch_size},
            {"type": "int", "name": "pooling_factor", "value": pooling_factor},
            {"type": "bool", "name": "weighted", "value": weighted},
            {"type": "str", "name": "weights_precision", "value": weights_precision},
        ],
        {"optimizer": {"type": "str", "value": optimizer}},
    )

    input_data_gen = op_config.input_data_generator()
    (input_args, input_kwargs) = input_data_gen.get_data(
        data_generator_config, run_options["device"]
    )

    return input_args[:len(fbgemm_input_args_indices(n))], input_kwargs # Discard weights if not needed


def build_torchscript_func(n):
    input_count = len(n.input_types)
    output_count = len(n.output_types)

    tmp = n.op_schema.split(') -> ')
    types = [item for item in tmp[0].split(' ') if ',' not in item][:-1]
    # print(n.name, n.id, types)
    types = [re.sub(r'\[[0-9]\]', '[]', t) for t in types] # e.g. int[2] -> int[]
    # print(n.name, n.id, types)
    input_types = ['Tensor' if 'Tensor(' in t else t for t in types if ('*)' not in t and '->' not in t)] # e.g. Tensor(float) -> Tensor; exception: aten::unbind(Tensor(a -> *) self, ...
    # print(n.name, n.id, input_types)
    input_types[0] = re.sub(r'^.*?\(', '', input_types[0]) # Strip the op name, e.g. aten::zeros(int[] -> int[]
    # print(n.name, n.id, input_types)
    output_types = tmp[-1].lstrip(' (').rstrip(')').split(', ') # e.g. "(Tensor, Tensor)" (str) -> [Tensor, Tensor] (list)
    output_types = ['Tensor' if 'Tensor(' in t else t for t in output_types]

    inputStr = """
        graph({}):
            {} = {}({})
            {}
            {}
    """.format(
        # Input arguments
        ", ".join(["%{}: {}".format(idx, t) for idx, t in enumerate(input_types)]),

        # Op
        "%output: {}".format(output_types[0] if output_count == 1 else "NoneType")
            if output_count <= 1
            else ", ".join(["%{}: {}".format(idx + input_count, t) for idx, t in enumerate(output_types)]),
        n.name,
        ", ".join(["%{}".format(idx) for idx in range(input_count)]),

        # Tuple handling
        "%output : ({}) = prim::TupleConstruct({})".format(
            ", ".join(["Tensor" for _ in range(output_count)]),
            ", ".join(["%{}".format(idx + input_count) for idx in range(output_count)]))
            if output_count > 1
            else "",

        # Return
        "return (%output)"
            if output_count >= 1
            else ""
    )
    # print(inputStr)
    # print("=============")
    graph = torch._C.parse_ir(inputStr)
    cu = torch._C.CompilationUnit()
    func = cu.create_function(n.name, graph)
    return func, output_count
