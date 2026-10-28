"""Inspect public Metal launch geometry of synthetic model-shaped operators.

Loads a measurement library into this process only. Instrumented runs are not
used for performance measurements. No model weights are loaded.
"""
import ctypes
import gc
import hashlib
import json
from pathlib import Path

import torch


@torch.inference_mode()
def main():
  root=Path('outputs/mac-gemm-tuning')
  if (root/'dispatches.json').exists():
    raise SystemExit('Refusing to overwrite dispatches.json')
  torch.set_num_threads(8)
  torch.manual_seed(218)
  torch.mps.set_per_process_memory_fraction(2*2**30/torch.mps.recommended_max_memory())
  probe=ctypes.CDLL(str((root/'metal_dispatch_probe.dylib').resolve()))
  probe.probe_start.argtypes=[ctypes.c_char_p]
  probe.probe_write.argtypes=[ctypes.c_char_p]
  assert probe.probe_install()==0
  report={'torch':torch.__version__,'torch_git_version':torch.version.git_version,
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[]}
  for batch in [1,2]:
    for name,n,k in [('gate_up',9216,2304),('down',2304,9216),
                     ('q',2048,2304),('kv',1024,2304),('o',2304,2048)]:
      x=torch.randn(batch,1024,k,device='mps')
      w=torch.randn(n,k,device='mps')/k**0.5
      for _ in range(2): y=torch.nn.functional.linear(x,w)
      torch.mps.synchronize()
      label=f'{name}_b{batch}'
      probe.probe_start(label.encode())
      y=torch.nn.functional.linear(x,w)
      torch.mps.synchronize()
      probe.probe_stop()
      expected=(x[0,0].cpu()[None]@w[:8].cpu().T).flatten()
      torch.testing.assert_close(y[0,0,:8].cpu(),expected,rtol=1e-4,atol=1e-4)
      report['cases'].append({'name':label,'M':batch*1024,'N':n,'K':k,
        'input_shape':list(x.shape),'input_stride':list(x.stride()),
        'weight_stride':list(w.stride()),'validated':True})
      del x,w,y;gc.collect();torch.mps.empty_cache()
  assert probe.probe_write(str((root/'dispatches.json').resolve()).encode())==0
  report['status']='complete'
  (root/'dispatch-inputs.json').write_text(json.dumps(report,indent=2)+'\n')
  print('Captured ten validated model-shaped projection calls.')


if __name__=='__main__':main()
