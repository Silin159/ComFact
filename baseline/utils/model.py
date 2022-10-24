import torch
# import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)


def run_batch_linking(args, model, batch, tokenizer=None):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, lm_labels, labels = batch

    if args.model_name_or_path.startswith("bert"):
        model_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=labels)
    else:
        model_outputs = model(input_ids=input_ids, labels=labels)

    cls_loss = model_outputs[0]
    cls_logits = model_outputs[1]
    # cls_loss = model_outputs.loss
    # cls_logits = model_outputs.logits
    lm_logits = None

    return cls_loss, lm_logits, cls_logits, labels
