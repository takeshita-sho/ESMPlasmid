Epoch 1:   0%|          | 0/2865 [00:00<?, ?it/s]Epoch 1:   0%|          | 0/2865 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 146, in <module>
    output = model(sequences,repr_layers=[1]) # feed in tokenized sequences, repr layer
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/nfshomes/stakeshi/esm/esm/model/esm2.py", line 85, in forward
    x = self.embed_scale * self.embed_tokens(tokens)
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/nn/modules/sparse.py", line 160, in forward
    self.norm_type, self.scale_grad_by_freq, self.sparse)
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/nn/functional.py", line 2199, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument index in method wrapper__index_select)
