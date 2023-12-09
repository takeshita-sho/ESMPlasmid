Epoch 1:   0%|          | 0/11458 [00:00<?, ?it/s]Epoch 1:   0%|          | 0/11458 [00:01<?, ?it/s]
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 81, in <module>
    output = model(sequences,repr_layers=[1]) # feed in tokenized sequences, repr layer
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/miniconda3/envs/esmfold/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/miniconda3/envs/esmfold/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/esm/esm/model/esm2.py", line 116, in forward
    x, attn = layer(
              ^^^^^^
  File "/nfshomes/stakeshi/miniconda3/envs/esmfold/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/miniconda3/envs/esmfold/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/esm/esm/modules.py", line 125, in forward
    x, attn = self.self_attn(
              ^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/miniconda3/envs/esmfold/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/miniconda3/envs/esmfold/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfshomes/stakeshi/esm/esm/multihead_attention.py", line 357, in forward
    attn_weights = torch.bmm(q, k.transpose(1, 2))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [enforce fail at alloc_cpu.cpp:83] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 2229980729344 bytes. Error code 12 (Cannot allocate memory)
