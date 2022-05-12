# N.B. Exgr utils required. Integration to Pytorch WIP.
import torch
from exec_graph_utils import NodeType
import io, json


# TODO: Add all torch dtypes to here

TORCH_DTYPES_RNG = {
    "int8": (torch.int8, torch.ones),
    "half": (torch.half, torch.ones),
    "int": (torch.int, torch.ones),
    "long": (torch.int64, torch.ones),
    "long int": (torch.int64, torch.ones),
    "float": (torch.float32, torch.randn),
    "double": (torch.float64, torch.randn)
}


YES_OPS = {
    "aten::convolution_backward", # Not the lowest BW aten
    "aten::max_pool2d_with_indices_backward", # ...
    "aten::nll_loss_backward", # ...
    "aten::_adaptive_avg_pool2d_backward", # ...

    "aten::max_pool2d_with_indices", # More outputs than its parent op
    "aten::log_softmax", # Together with nll_loss_forward
    "aten::nll_loss_forward", # More outputs than its parent op
    "aten::native_dropout", # More outputs than its parent op
}


NO_OPS = {
    "aten::max_pool2d", # Has only one output explicitly but its child op implicitly has two (aten::max_pool2d_with_indices)
    "aten::dropout", # ... (aten::native_dropout)
    "aten::cross_entropy_loss", # ... (aten::nll_loss_forward)
    "aten::nll_loss_nd", # ... (aten::nll_loss_forward)
}


def is_tensor(n, ip):
    return isinstance(ip, int) and 'Tensor' in n.input_types[ip]


def is_op(node):
    return (node.type == NodeType.OPERATOR and (node.parent is not None and node.parent.type != NodeType.OPERATOR))


def is_lowest_level_aten(op):
    return op.name.startswith("aten::") and (
        not op.children or op.name == "aten::addmm"
    )


def has_backward_parent(op):
    if not op.parent or op.parent.id == op.id:
        return False
    if is_backward(op):
        return True
    return has_backward_parent(op.parent)


def is_backward(op):
    return "autograd::engine::evaluate_function: " in op.name or \
            "Optimizer" in op.name


def is_backward_aten(op):
    return op.name.startswith("aten::") and \
            not op.children and \
            has_backward_parent(op)


def special_treatment(op):
    if op.name in YES_OPS:
        return True
    elif op.name in NO_OPS:
        return False
    return None


def is_qualified(op):
    sp = special_treatment(op)
    if sp is not None:
        return sp
    if op.get_parent_by_name(YES_OPS): # Discard all ops under any YES_OP
        return False
    return (is_backward_aten(op)) or (is_op(op) and not is_backward(op))


def trace_handler(prof):
    # print(prof.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

def another_trace_handler(prof):
    # print(prof.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("/tmp/test_trace_replay.json")


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
