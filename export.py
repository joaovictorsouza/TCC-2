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
from transformers import BertForPreTraining

TORCH_LOGS="+dynamic"

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
    bert = BertForPreTraining.from_pretrained('neuralmind/bert-base-portuguese-cased')

    example_inputs = prepare_inputs("Oi tudo bem?", tokenizer),
    print(example_inputs)

        
    # Definir as dimensões dinâmicas (corrigir a dimensão correta com base na forma do tensor)
    dynamic_shape = ({1: torch.export.Dim("token_dim", min=1, max=512)},)

    # Substitua o antigo capture_pre_autograd_graph por torch.export
    # m = torch.export(model, example_inputs, dynamic_shapes=dynamic_shape)
    # m = torch.jit.script(model, example_inputs)

    # # Salvar o modelo exportado
    # torch.save(m, "model_scripted.pt")

    m = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shape)
    torch.export.save(m, "model_exported.pt")
