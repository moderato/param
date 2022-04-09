import torch
import re
from exec_graph_utils import NodeType


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
    return "autograd::engine::evaluate_function: " in op.name


def is_backward_aten(op):
    return op.name.startswith("aten::") and \
            not op.children and \
            has_backward_parent(op)


# Backup starts here
# CONSIDER = ["aten::linear", "AddmmBackward", "aten::bmm", "BmmBackward0", "aten::matmul", "MmBackward", \
#                 "aten::conv2d", "CudnnConvolutionBackward", \
#                 "LookupFunction", "LookupFunctionBackward", \
#                 "aten::batch_norm", "CudnnBatchNormBackward", \
#                 "aten::index", "IndexBackward", \
#                 "aten::relu", "aten::relu_", "ReluBackward0", "ReluBackward1", \
#                 "aten::sigmoid", "SigmoidBackward", \
#                 "aten::binary_cross_entropy", "BinaryCrossEntropyBackward", \
#                 "aten::mse_loss", "MseLossBackward", \
#                 "aten::avg_pool2d", "AvgPool2D", \
#                 "aten::max_pool2d", "MaxPool2DWithIndicesBackward", \
#                 "aten::add", "aten::add_", "aten::__and__", \
#                 "aten::mul", "aten::div", "aten::sum", "SummBackward0p"\
#                 "aten::cat", "aten::to", "aten::ones_like", \
#                 "torch::autograd::AccumulateGrad", "Optimizer.step#SGD.step", "Optimizer.zero_grad#SGD.zero_grad"]


# COMMS = ["nccl:all_to_all", "nccl:all_reduce"]


# def op_name_in_list(op, lst):
#     if op.name in lst:
#         return True
#     if is_backward(op):
#         bw_truncated_name = op.name.split("autograd::engine::evaluate_function: ")[-1]
#         return bw_truncated_name in lst or \
#                 bw_truncated_name[:-1] in lst # Truncate trailing 0/1
#     return False


# # def to_consider(op):
# #     return op_name_in_list(op, CONSIDER)


# # def to_skip(op):
# #     return op_name_in_list(op, SKIP)

# # (keyword_arguments, positional argument count)
# SPECIAL_KWARGS_MAP = {
#     "aten::batch_norm": (["weight", "bias", "training", "momentum", "eps"], 3),
#     "aten::cross_entropy_loss": ([
#         "weight", "size_average", "ignore_index", \
#         "reduce", "reduction", "label_smoothing"
#     ], 2),
#     "aten::_reshape_alias": ([], 2),
#     "aten::view": ([], -1),
#     "aten::copy_": ([], -1),
#     "aten::_log_softmax_backward_data": (["dim", "input_dtype"], 2)
# }


# ENUM_MAP = {
#     "input_dtype": (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, \
#                         torch.float16, torch.float32, torch.float64
#                     ),
#     "dtype": (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, \
#                 torch.float16, torch.float32, torch.float64
#             ),
#     "memory_format": (torch.contiguous_format, torch.preserve_format, torch.channels_last, torch.channels_last_3d),
#     "layout": (torch.strided, torch.sparse_coo, torch.sparse_csr)
# }


# def get_pytorch_kwargs(f):
#     s = f.__doc__.split('\n')[1].lstrip(' ')
#     split_equal = s.split('=')[:-1]
#     split_space = split_equal[0].split(' ')[:-1]
#     positional_argument_count = len(split_space)
#     if '*' in split_space[-1]:
#         positional_argument_count -= 1
#     # print(split_space, positional_argument_count)
#     l = []
#     for ss in split_equal:
#         l.append(ss.split(' ')[-1])
#     return l, positional_argument_count


# MODULE_FUNCTION_MAP = {
#     "aten::linear": torch.nn.functional.linear,
#     "aten::mm": torch.mm,
#     "aten::matmul": torch.addmm,
#     "aten::bmm": torch.bmm,
#     "aten::conv2d": torch.nn.functional.conv2d,
#     "aten::batch_norm": torch.nn.functional.batch_norm,
#     "aten::max_pool2d": torch.nn.functional.max_pool2d,
#     "aten::avg_pool2d": torch.nn.functional.avg_pool2d,
#     "aten::cross_entropy_loss": torch.nn.functional.cross_entropy,
#     "aten::add": torch.add,
#     "aten::sub": torch.sub,
#     "aten::mul": torch.mul,
#     "aten::div": torch.div,
#     "aten::neg": torch.neg,
#     "aten::sum": torch.sum,
#     "aten::ones_like": torch.ones_like,
#     "aten::empty": torch.empty,
#     "aten::view": lambda x, y: x.view(y),
#     "aten::empty_strided": torch.empty_strided,
#     "aten::as_strided": torch.as_strided,
#     "aten::_reshape_alias": torch.reshape,
#     "aten::copy_": lambda x, y, z: x.copy_(y, z),
#     "aten::_log_softmax_backward_data": torch._log_softmax_backward_data,
# }


# # {
# #   op_name: [list of keywords, positional_argument_count]
# # }
# FUNCTION_KEYWORDS = {
#     k: get_pytorch_kwargs(MODULE_FUNCTION_MAP[k]) 
#         if k not in SPECIAL_KWARGS_MAP.keys() else SPECIAL_KWARGS_MAP[k]
#         for k, _ in MODULE_FUNCTION_MAP.items()
# }


# TYPES_CONVERSION  = [
#     ("Tensor\(", "Tensor"),
#     ("Int", "int"),
#     ("GeneralList\[Int*", "int[]"),
#     ("Float", "float"),
#     ("GeneralList\[Float*", "float[]"),
#     ("Double", "double"),
#     ("GeneralList\[Double*", "double[]"),
#     ("Bool", "bool"),
# ]

# def lookup(s, lookups):
#     for pattern, value in lookups:
#         if re.search(pattern, s):
#             return value
#     return None


# from pprint import pprint
# pprint(FUNCTION_KEYWORDS)
