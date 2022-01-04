import os
import sys
import onnx
import argparse
import subprocess

"""
This script is used to check models in https://github.com/onnx/models.
It automatically downloads a model from onnx/models, compiles the model by
using onnx-mlir, and deletes the model. 

Note:
    - This script must be invoked from the root folder of https://github.com/onnx/models.
    - This script requires git-lfs to download models. Please follow the instruction here to install git-lfs: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage 
    - Environment variable ONNX_MLIR_HOME is needed to find onnx-mlir.
"""

if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found.")

ONNX_MLIR_EXENAME = "onnx-mlir"
if sys.platform == "win32":
    ONNX_MLIR_EXENAME = "onnx-mlir.exe"

ONNX_MLIR = os.path.join(os.environ['ONNX_MLIR_HOME'], "bin",
                         ONNX_MLIR_EXENAME)

VERBOSE = os.environ.get('VERBOSE', False)

# Keep this list synced with onnx-mlir.
onnx_mlir_ops = set([name.lower() for name in [
    "Abs",
    "Acos",
    "Acosh",
    # Adagrad
    # Adam
    "Add",
    "And",
    "Argmax",
    # Argmin
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    # AveragePool: same_upper/lower dyn padding-shapes not supported.
    "AveragePool",
    # BatchNormalization (test mode)
    "BatchNormalization",
    # Bitshift left/right
    "Cast",
    "Ceil",
    # Celu
    "Clip",
    "Compress",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Conv",
    # ConvInteger
    # ConvTranspose
    "Cos",
    "CumSum",
    "DepthToSpace",
    # DequatizeLinear
    # Det
    "Div",
    "Dropout",
    # DynamicQuantizeLinear
    # Edge
    # EinSum
    "Elu",
    "Equal",
    "Erf",
    "Exp",
    "Expand",
    # Eyelike
    "Flatten",
    "Floor",
    "Gather",
    "Gemm",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "Greater",
    "GreaterOrEqual",
    "GRU",
    "HardMax",
    "HardSigmoid",
    "Identity",
    "InstanceNormalization",
    # Is Inf Neg/Pos
    # Is Nan
    "LeakyRelu",
    "Less",
    "LessOrEqual",
    "Log",
    "LogSoftmax",
    "Loop",
    "LRN",
    "LSTM",
    "MatMul",
    # Matmul Integer
    "Max",
    "MaxPool",
    "Mean",
    "Min",
    "Mod",
    # Momentum
    "Mul",
    # Multinomial (NMV)
    "Neg",
    # Negative Log Likelihood Loss
    "NonMaxSuppression",
    "NonZero",
    "Not",
    "OneHot",
    "Or",
    "Pad",
    "Pow",
    "PRelu",
    # QLinear Conv
    # QLinear Matmul
    # Quantize Linear
    "Range",
    "Reciprocal",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
    "Relu",
    "Reshape",
    "Resize",
    "ReverseSequence",
    "RNN",
    # Roi Align
    "Round",
    "Scan",
    # Scatter Element
    "Selu",
    "Shape",
    # Shrink
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Size",
    "Slice",
    "Softmax",
    "Softplus",
    "Softsign",
    "Split",
    "Sqrt",
    "Squeeze",
    # Str Normalizer
    "Sub",
    "Sum",
    "Tan",
    "Tanh",
    # Tfdf Vectorizer
    # Threshold Relu
    "Tile",
    "TopK",
    # Training Dropout
    "Transpose",
    # Unique
    "Unsqueeze",
    "Upsample",
    "Where",
    "Xor",
]])

# Deprecated models according to: https://github.com/onnx/models/pull/389
deprecated_models = {
    "mnist-1.onnx",
    "bvlcalexnet-3.onnx",
    "caffenet-3.onnx",
    "densenet-3.onnx",
    "inception-v1-3.onnx",
    "inception-v2-3.onnx",
    "rcnn-ilsvrc13-3.onnx",
    "resnet50-caffe2-v1-3.onnx",
    "shufflenet-3.onnx",
    "zfnet512-3.onnx",
    "vgg19-caffe2-3.onnx",
    "emotion-ferplus-2.onnx",
}

# Compiler flags for a specific model. 
compiler_flags = {
    #"gpt2-10.onnx": ["--repeatOnnxTransform=1"],
    #"gpt2-lm-head-10.onnx": ["--repeatOnnxTransform=1"],
    #"t5-encoder-12.onnx": ["--repeatOnnxTransform=1"],
    #"bertsquad-8.onnx": ["--repeatOnnxTransform=1"],
    #"bertsquad-10.onnx": ["--repeatOnnxTransform=1"],
    #"roberta-sequence-classification-9.onnx": ["--repeatOnnxTransform=1"],
    #"roberta-base-11.onnx": ["--repeatOnnxTransform=1"],
}

NEW_LINE = "\n"
NODE_DELIMITER = "\n"
INPUT_DELIMITER = ", "
OUTPUT_DELIMITER = ", "
ATTRIBUTE_DELIMITER = ", "
LIST_DELIMITER = ", "


class ModelReader:
    def __init__(self, model_path):
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self.op_name_set = set()
        self.graph_in_text = ""

    def run(self):
        self.graph_in_text = self.readGraph(0)

    def printGeneralInfo(self):
        print("ir_version: {}".format(self.model.ir_version))
        print("opset_import: {}".format(self.model.opset_import))
        print("producer_name: {}".format(self.model.producer_name))
        print("producer_version: {}".format(self.model.producer_version))
        print("domain: {}".format(self.model.domain))
        print("model_version: {}".format(self.model.model_version))
        print("doc_string: {}\n".format(self.model.doc_string))

    def readGraph(self, indent):
        statement = ' '*indent + self.graph.name + " {"
        statement += NEW_LINE
        for node in self.graph.node:
            statement += self.readNode(node, 2) + NODE_DELIMITER
        statement += ' '*indent + "}"
        return statement

    def readNode(self, node, indent=0):
        statement = ' ' * indent
        # output
        for output in node.output:
            statement += output + OUTPUT_DELIMITER
        statement = statement[:-len(OUTPUT_DELIMITER)]

        statement += " = "
        statement += node.op_type
        self.op_name_set.add(node.op_type.lower())

        # input
        statement += "("
        for input in node.input:
            if (input):
                statement += input + INPUT_DELIMITER
            else:
                statement += "None" + INPUT_DELIMITER
        statement = statement[:-len(INPUT_DELIMITER)]
        statement += ")"

        # attribute
        if (len(node.attribute) > 0):
          statement += " {"
          for attr in node.attribute:
              statement += self.readAttribute(attr) + ATTRIBUTE_DELIMITER
          statement = statement[:-len(ATTRIBUTE_DELIMITER)]
          statement += "}"

        return statement

    def readAttribute(self, attr, indent=0):
        statement = ' ' * indent
        statement += attr.name
        statement += " : "
        if (attr.type == 1):
          statement += str(attr.f)
        if (attr.type == 2):
          statement += str(attr.i)
        if (attr.type == 3):
          statement += attr.s.decode("utf-8")
        if (attr.type == 6):
          statement += str(attr.floats)
        if (attr.type == 7):
          statement += str(attr.ints)
        if (attr.type == 8):
            statement += "["
            for s in attr.strings:
                statement += s.decode("utf-8") + LIST_DELIMITER
            statement = statement[:-len(LIST_DELIMITER)]
            statement += "]"
        return statement


def execute_commands(cmds):
    if (VERBOSE):
        print(cmds)
    out = subprocess.Popen(cmds, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    stdout, stderr = out.communicate()
    if stderr:
        return (False, stderr.decode("utf-8"))
    else:
        return (True, stdout.decode("utf-8"))


def execute_commands_to_file(cmds, ofile):
    if (VERBOSE):
        print(cmds)
    with open(ofile, 'w') as output:
        server = subprocess.Popen(
            cmds, stdout=output, stderr=subprocess.STDOUT)
        stdout, stderr = server.communicate()


FIND_MODEL_PATHS_CMD = ['find', '.', '-type', 'f', '-name', '*.onnx']
# git lfs pull --include="${onnx_model}" --exclude=""
PULL_CMD = ['git', 'lfs', 'pull', '--exclude=\"\"']
# git lfs pointer --file = "${onnx_model}" > ${onnx_model}.pt
CLEAN_CMD = ['git', 'lfs', 'pointer']
CHECKOUT_CMD = ['git', 'checkout', '-f', 'master']
RM_CMD = ['rm']
MV_CMD = ['mv']


def pull_and_get_ops_from_model_zoo(count=-1, single_model=None, keep_model=False):
    _, model_paths = execute_commands(FIND_MODEL_PATHS_CMD)
    model_paths = model_paths.split('\n')
    # Remove empty paths and prune '._' in a path.
    model_paths = [path[2:] for path in model_paths if path]
    model_names = [path.split('/')[-1] for path in model_paths]
    deprecated_names = set(model_names).intersection(deprecated_models)

    print('\n')
    deprecated_msg = ""
    if (len(deprecated_names) != 0):
        deprecated_msg = "where " + \
            str(len(deprecated_names)) + \
            " models are deprecated (using very old opsets, e.g. <= 3)"
    print("# There are {} models in the ONNX model zoo {}".format(
        len(model_paths), deprecated_msg))
    print("See https://github.com/onnx/models/pull/389",
          "for a list of deprecated models\n")

    # Read each model in the zoo and collect ops in the model.
    model_to_ops_dict = {}
    i = 0
    for path in model_paths:
        model_name = path.split('/')[-1]
        # Ignore deprecated models.
        if model_name in deprecated_models:
            continue
        # If process only a single given model.
        if single_model and single_model != model_name:
            continue
        i += 1
        print('[{}] download and compile'.format(i), path)
        # pull the model.
        pull_cmd = PULL_CMD + ['--include={}'.format(path)]
        execute_commands(pull_cmd)
        # read the set of ops in the model.
        model_reader = ModelReader(path)
        model_reader.run()
        # try to compile.
        options = []
        if model_name in compiler_flags:
            options = compiler_flags[model_name]
        isCompilable, msg = execute_commands([ONNX_MLIR, path] + options)
        if isCompilable:
            # delete the generated .so file when the model is compilable.
            execute_commands(RM_CMD + [path[:-4] + 'so'])

        # store the set to the dict.
        model_to_ops_dict[model_name.lower()] = (
            model_reader.op_name_set, isCompilable, '' if isCompilable else msg)
        if not keep_model:
            # remove the model to save the storage space.
            clean_cmd = CLEAN_CMD + ['--file={}'.format(path)]
            execute_commands_to_file(clean_cmd, '{}.pt'.format(path))
            execute_commands(RM_CMD + [path])
            execute_commands(MV_CMD + ['{}.pt'.format(path), path])
            execute_commands(CHECKOUT_CMD)
        if i == count:
            break
    return model_to_ops_dict


'''Analyze a model and print out information in markdown format.
'''


def analyze(model_to_ops_dict):
    I = '|'
    print("\n")
    print("# ONNX models and their ops\n")
    print(I, 'ONNX model', I, 'Ops in the model',
          I, 'Ops not supported in onnx-mlir',
          I, 'Compilable with onnx-mlir', I)
    print(I, '-----', I, '-----', I, '-----', I, '-----', I)
    number_of_supported_models = 0
    number_of_compiled_models = 0
    for key in sorted(model_to_ops_dict):
        model_ops, isCompilable, msg = model_to_ops_dict[key]
        diff = model_ops - onnx_mlir_ops
        if isCompilable:
            number_of_compiled_models += 1
        if (diff == set()):
            number_of_supported_models += 1
        isSupported = "supported" if (diff == set()) else "not supported"
        compilable = 'succeeded' if isCompilable else msg.replace('\n', '<br>')
        diff = "{}" if (diff == set()) else diff
        print_key = key
        if (key in compiler_flags):
            print_key += " {}".format(compiler_flags[key])
        print(I, print_key, I, model_ops, I, diff, I, compilable, I)

    print("\n")
    print("Looks like ONNX-MLIR supports {} models,".format(number_of_supported_models),
          "of which {} models can be really compiled".format(
              number_of_compiled_models),
          "and {} models failed to compile".format(
              number_of_supported_models - number_of_compiled_models)
          )

    # Do analyses.
    all_ops = set()
    for key in model_to_ops_dict:
        model_ops, _, _ = model_to_ops_dict[key]
        all_ops = all_ops.union(model_ops)
    # max indent to print op name.
    max_indent = max([len(name) for name in all_ops]) + 5
    # Count occurrence of an op in models.
    op_count_dict = {}
    for op in all_ops:
        op_count_dict[op] = 0
        for key in model_to_ops_dict:
            if op in model_to_ops_dict[key][0]:
                op_count_dict[op] += 1
    print("\n")
    print("# Count the number of models in which an op is used (sorted in the decreasing order):\n")
    # sort by value
    xs = sorted(op_count_dict.items(), key=lambda item: item[1], reverse=True)
    header1 = 'Operator name'
    header2 = 'Count'
    header3 = 'Supported in onnx-mlir'
    print(I, header1 + ' '*(max_indent - len(header1)),
          I, header2,
          I, header3, I)
    print(I, '-'*(max_indent),
          I, '-'*len(header2),
          I, '-'*len(header3), I)
    for op_name, count in xs:
        isSupported = "supported" if op_name in onnx_mlir_ops else "not supported"
        print(I, op_name + ' ' * (max_indent - len(op_name)),
              I, str(count) + ' ' * (len(header2) - len(str(count))),
              I, isSupported + ' ' * (len(header3) - len(isSupported)), I)
    print('\n')


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-m', '--model', help="onnx model")
    group.add_argument('-z', '--model_zoo', action='store_true',
                       help="analyze ops in the ONNX model zoo. "
                       "Need to run this script from the root folder of "
                       "onnx/models repository. Take about 2 hours. "
                       "Output is in markdown format")
    parser.add_argument('-s', '--single_model',
                        help="Only process a single model in the ONNX model zoo")
    parser.add_argument('-c', '--model_count', default=-1,
                        help="the number of models in the zoo to process. "
                        "All model by default")
    parser.add_argument('-p', '--print_set_of_ops', action='store_true',
                        help="print a set of operations only")
    parser.add_argument('-k', '--keep_model', action='store_true',
                        help="keep the downloaded model or not")
    args = parser.parse_args()

    if args.model:
        model_reader = ModelReader(args.model)
        model_reader.run()
        if args.print_set_of_ops:
            print(model_reader.op_name_set)
        else:
            model_reader.printGeneralInfo()
            print("==========Computation graph==========")
            print(model_reader.graph_in_text)
            print("==========End of Computation graph==========")

    if args.model_zoo:
        print("# ONNX-MLIR supports {} ONNX ops\n".format(len(onnx_mlir_ops)))
        print(sorted(onnx_mlir_ops))
        data = pull_and_get_ops_from_model_zoo(
            int(args.model_count), args.single_model, args.keep_model)
        analyze(data)


if __name__ == "__main__":
    main()
