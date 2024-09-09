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
from torch.export import Dim
import ai_edge_torch



TORCH_LOGS="+dynamic"

def adjust_to_multiple_of_8(tokens):
    while len(tokens) % 8 != 0:
        tokens.append(0)  # Adiciona padding (ou outro token adequado)
    return tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint file path")
    args = parser.parse_args()

    print("Wait... loading model")
    ckpt = args.ckpt

    model = Net(hp.n_classes)
    model = model
    ckpt = torch.load(ckpt)
    # model.load_state_dict(ckpt)

    # ckpt = OrderedDict([(k.replace("module.", "").replace("LayerNorm.weight", "LayerNorm.gamma").replace("LayerNorm.bias", "LayerNorm.beta"), v) for k, v in ckpt.items()])
    ckpt = OrderedDict([(k.replace("module.", ""), v) for k, v in ckpt.items()])
    model.load_state_dict(ckpt)

    idx2phr = pickle.load(open(hp.idx2phr, 'rb'))

    tokenizer = BertTokenizer.from_pretrained('adalbertojunior/distilbert-portuguese-cased', do_lower_case=True)
    bert = BertForPreTraining.from_pretrained('neuralmind/bert-base-portuguese-cased')

    # example_inputs = prepare_inputs("Oi tudo bem?", tokenizer),
    # print(example_inputs)

        
    # Definir as dimensões dinâmicas (corrigir a dimensão correta com base na forma do tensor)
    # dynamic_shape = ({1: torch.export.Dim("token_dim", min=1, max=512)},)

    # Substitua o antigo capture_pre_autograd_graph por torch.export
    # m = torch.export(model, example_inputs, dynamic_shapes=dynamic_shape)
    # m = torch.jit.script(model, example_inputs)

    # # Salvar o modelo exportado
    # torch.save(m, "model_scripted.pt")
    # Tokenizing input text
    text = "[CLS] Quem é você ? [SEP] Não gosto do seu jeito [SEP]"
    tokenized_text = tokenizer.tokenize(text)

    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    indexed_tokens = adjust_to_multiple_of_8(indexed_tokens)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens[:512]])
    segments_tensors = torch.tensor([segments_ids[:512]])
    dummy_input = (tokens_tensor, segments_tensors)

    # _token_dim = Dim('_token_dim', min=1, max=64)  # Define o intervalo dinâmico como múltiplo de 8
    # dim_batch = 1

    # # Garante que token_dim é múltiplo de 8
    # dynamic_shapes = {
    #     "x": {0: dim_batch, 1: 8 * _token_dim},       # token_dim como múltiplo de 8
    #     "attention_mask": {0: dim_batch, 1: 8 * _token_dim}  # Mesmo ajuste para attention_mask
    # }
    
    # traced_model = torch.export.export(model, (tokens_tensor, segments_tensors), dynamic_shapes=dynamic_shapes)
    edge_model = ai_edge_torch.convert(model.eval(), dummy_input)
    edge_model.export('bert.tflite')

    # torch.export.save(traced_model, "exported_bert.pt")
