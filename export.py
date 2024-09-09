import argparse
from test import prepare_inputs
from model import Net
import torch
from collections import OrderedDict
from hparams import hp
import pickle
from executorch.exir import EdgeCompileConfig, to_edge
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch._export import capture_pre_autograd_graph
from torch.export import export
from transformers import BertTokenizer  # Or BertTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint file path")
    args = parser.parse_args()

    print("Wait... loading model")
    ckpt = args.ckpt

    model = Net(hp.n_classes)
    model = model.cuda()
    ckpt = torch.load(ckpt)
    # model.load_state_dict(ckpt)

    # ckpt = OrderedDict([(k.replace("module.", "").replace("LayerNorm.weight", "LayerNorm.gamma").replace("LayerNorm.bias", "LayerNorm.beta"), v) for k, v in ckpt.items()])
    ckpt = OrderedDict([(k.replace("module.", ""), v) for k, v in ckpt.items()])
    model.load_state_dict(ckpt)

    idx2phr = pickle.load(open(hp.idx2phr, 'rb'))

    tokenizer = BertTokenizer.from_pretrained('adalbertojunior/distilbert-portuguese-cased', do_lower_case=True)

    example_inputs = prepare_inputs("Você pode me ajudar?", tokenizer)

    dynamic_shape = (
    {1: torch.export.Dim("token_dim", max=model.config.block_size)},
    )

    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        m = capture_pre_autograd_graph(model, example_inputs, dynamic_shapes=dynamic_shape)
        traced_model = export(m, example_inputs, dynamic_shapes=dynamic_shape)

    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    edge_manager = to_edge(traced_model,  compile_config=edge_config)
    et_program = edge_manager.to_executorch()

    # Save the ExecuTorch program to a file.
    with open("tcc2.pte", "wb") as file:
        file.write(et_program.buffer)