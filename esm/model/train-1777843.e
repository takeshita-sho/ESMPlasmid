/nfshomes/stakeshi/miniconda3/envs/esmfold/lib/python3.11/site-packages/torch/nn/modules/transformer.py:20: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /opt/conda/conda-bld/pytorch_1699449201450/work/torch/csrc/utils/tensor_numpy.cpp:84.)
  device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
Epoch 1:   0%|          | 0/11458 [00:00<?, ?it/s]Epoch 1:   0%|          | 0/11458 [00:00<?, ?it/s]
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
  File "/nfshomes/stakeshi/esm/esm/model/esm2.py", line 82, in forward
    assert tokens.ndim == 2
           ^^^^^^^^^^^^^^^^
AssertionError
