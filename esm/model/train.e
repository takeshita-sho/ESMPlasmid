/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/distributed/launch.py:186: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  FutureWarning,
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 24, in <module>
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 24, in <module>
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 24, in <module>
    print("Using: " + device)
TypeError: can only concatenate str (not "torch.device") to str
    print("Using: " + device)
TypeError: can only concatenate str (not "torch.device") to str
    print("Using: " + device)
TypeError: can only concatenate str (not "torch.device") to str
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 24, in <module>
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 24, in <module>
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 24, in <module>
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 24, in <module>
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 24, in <module>
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 24, in <module>
Traceback (most recent call last):
  File "/nfshomes/stakeshi/esm/esm/model/train.py", line 24, in <module>
    print("Using: " + device)
TypeError: can only concatenate str (not "torch.device") to str
    print("Using: " + device)
TypeError: can only concatenate str (not "torch.device") to str
    print("Using: " + device)
TypeError: can only concatenate str (not "torch.device") to str
    print("Using: " + device)
TypeError: can only concatenate str (not "torch.device") to str
    print("Using: " + device)
TypeError: can only concatenate str (not "torch.device") to str
    print("Using: " + device)
TypeError: can only concatenate str (not "torch.device") to str
    print("Using: " + device)
TypeError: can only concatenate str (not "torch.device") to str
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 49602) of binary: /nfshomes/stakeshi/anaconda3/envs/esmfold/bin/python3
Traceback (most recent call last):
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/distributed/launch.py", line 193, in <module>
    main()
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/distributed/launch.py", line 189, in main
    launch(args)
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/distributed/launch.py", line 174, in launch
    run(args)
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/distributed/run.py", line 755, in run
    )(*cmd_args)
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/distributed/launcher/api.py", line 131, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/nfshomes/stakeshi/anaconda3/envs/esmfold/lib/python3.7/site-packages/torch/distributed/launcher/api.py", line 247, in launch_agent
    failures=result.failures,
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
/nfshomes/stakeshi/esm/esm/model/train.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2023-10-27_14:11:31
  host      : cbcb27.umiacs.umd.edu
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 49603)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2023-10-27_14:11:31
  host      : cbcb27.umiacs.umd.edu
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 49604)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2023-10-27_14:11:31
  host      : cbcb27.umiacs.umd.edu
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 49605)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[4]:
  time      : 2023-10-27_14:11:31
  host      : cbcb27.umiacs.umd.edu
  rank      : 4 (local_rank: 4)
  exitcode  : 1 (pid: 49606)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[5]:
  time      : 2023-10-27_14:11:31
  host      : cbcb27.umiacs.umd.edu
  rank      : 5 (local_rank: 5)
  exitcode  : 1 (pid: 49607)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[6]:
  time      : 2023-10-27_14:11:31
  host      : cbcb27.umiacs.umd.edu
  rank      : 6 (local_rank: 6)
  exitcode  : 1 (pid: 49608)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[7]:
  time      : 2023-10-27_14:11:31
  host      : cbcb27.umiacs.umd.edu
  rank      : 7 (local_rank: 7)
  exitcode  : 1 (pid: 49609)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[8]:
  time      : 2023-10-27_14:11:31
  host      : cbcb27.umiacs.umd.edu
  rank      : 8 (local_rank: 8)
  exitcode  : 1 (pid: 49610)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[9]:
  time      : 2023-10-27_14:11:31
  host      : cbcb27.umiacs.umd.edu
  rank      : 9 (local_rank: 9)
  exitcode  : 1 (pid: 49611)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-10-27_14:11:31
  host      : cbcb27.umiacs.umd.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 49602)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
