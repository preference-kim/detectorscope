"""Bounded FP32 operator calibration for the streamed Gemma-2 shapes.

No backbone is loaded. Results report useful FLOPs and compulsory tensor bytes,
not hardware instruction/DRAM counters. Single-call latency and queued throughput
are measured separately; input allocation, compilation and validation are untimed.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

import torch
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm, apply_rotary_pos_emb

from benchmark_mac import write_json


@torch.inference_mode()
def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', default='outputs/mac-roofline/operators.json')
  args = parser.parse_args()
  output = Path(args.output)
  if output.exists():
    parser.error('Choose a fresh output file to preserve previous measurements')
  torch.set_num_threads(8)
  torch.manual_seed(892)
  torch.mps.set_per_process_memory_fraction(min(1.0, 2*2**30/torch.mps.recommended_max_memory()))
  report = {'status': 'running', 'torch': torch.__version__,
            'torch_git_version': torch.version.git_version, 'dtype': 'float32',
            'cpu_threads': torch.get_num_threads(), 'runs': [],
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

  def bench(name, fn, *, flops=None, bytes_min=None, queued=8, metadata=None):
    for _ in range(2):
      value = fn()
    torch.mps.synchronize()
    rows = []
    for repeat in range(7):
      # Alternate timing style to avoid always warming one style with the other.
      for inner in ([1, queued] if repeat % 2 else [queued, 1]):
        torch.mps.synchronize()
        started = time.perf_counter()
        for _ in range(inner):
          value = fn()
        torch.mps.synchronize()
        rows.append({'repeat': repeat, 'dispatches': inner,
                     'seconds': (time.perf_counter()-started)/inner})
    if isinstance(value, torch.Tensor):
      assert torch.isfinite(value.flatten()[:32]).all().item()
    item = {'name': name, 'flops': flops, 'minimum_tensor_bytes': bytes_min,
            'metadata': metadata or {}, 'samples': rows,
            'single_median_seconds': statistics.median(r['seconds'] for r in rows if r['dispatches']==1),
            'queued_median_seconds': statistics.median(r['seconds'] for r in rows if r['dispatches']==queued),
            'queued_dispatches': queued,
            'sampled_driver_bytes': torch.mps.driver_allocated_memory()}
    report['runs'].append(item)
    write_json(output, report)
    print(json.dumps({k:v for k,v in item.items() if k not in ('samples','metadata')}), flush=True)
    return value

  # Large square GEMM gives an empirical sustained-compute reference. It is
  # not labeled as the hardware's unpublished architectural peak FLOPS.
  for size in [1024, 2048, 4096]:
    x=torch.randn(size,size,device='mps'); w=torch.randn(size,size,device='mps')/(size**0.5)
    value=bench(f'gemm_square_{size}',lambda:torch.nn.functional.linear(x,w),
                flops=2*size**3,bytes_min=12*size**2,queued=4,
                metadata={'M':size,'N':size,'K':size,'input_rank':2})
    expected=(x[0].cpu()[None] @ w[:8].cpu().T).flatten()
    torch.testing.assert_close(value[0,:8].cpu(),expected,rtol=1e-4,atol=1e-4)
    del x,w,value;gc.collect();torch.mps.empty_cache()

  projections={'gate_up':(9216,2304),'down':(2304,9216),
               'q':(2048,2304),'kv':(1024,2304),'o':(2304,2048)}
  for batch in [1,2]:
    m=batch*1024
    for label,(n,k) in projections.items():
      x=torch.randn(batch,1024,k,device='mps');w=torch.randn(n,k,device='mps')/(k**0.5)
      value=bench(f'{label}_b{batch}',lambda:torch.nn.functional.linear(x,w),
                  flops=2*m*n*k,bytes_min=4*(m*k+n*k+m*n),queued=4,
                  metadata={'M':m,'N':n,'K':k,'input_rank':3,'input_stride':list(x.stride())})
      expected=(x[0,0].cpu()[None] @ w[:8].cpu().T).flatten()
      torch.testing.assert_close(value[0,0,:8].cpu(),expected,rtol=1e-4,atol=1e-4)
      if batch==2 and label=='gate_up':
        bench('gate_up_b2_flattened',lambda:torch.nn.functional.linear(x.flatten(0,1),w),
              flops=2*m*n*k,bytes_min=4*(m*k+n*k+m*n),queued=4,
              metadata={'M':m,'N':n,'K':k,'input_rank':2})
      del x,w,value;gc.collect();torch.mps.empty_cache()

  for label,(n,k) in list(projections.items())[:2]:
    x=torch.randn(k,device='mps');w=torch.randn(n,k,device='mps')/(k**0.5)
    value=bench(f'gemv_{label}',lambda:torch.nn.functional.linear(x,w),
                flops=2*n*k,bytes_min=4*(k+n*k+n),queued=16,
                metadata={'M':1,'N':n,'K':k,'input_rank':1})
    torch.testing.assert_close(value[:8].cpu(),w[:8].cpu()@x.cpu(),rtol=1e-4,atol=1e-4)
    del x,w,value;gc.collect();torch.mps.empty_cache()

  for batch in [1,2]:
    m=batch*1024; h=2304; n=m*h
    x=torch.randn(batch,1024,h,device='mps');norm=Gemma2RMSNorm(h).eval().to('mps')
    bench(f'rmsnorm_b{batch}',lambda:norm(x),flops=4*n+2*m+h,
          bytes_min=4*(2*n+h),queued=16,
          metadata={'elements':n,'rows':m,'width':h,'rsqrt_count':m,
                    'unfused_tensor_traffic_estimate':4*(7*n+6*m+3*h)})
    for label,fn,bytes_min in [
        ('square',lambda:torch.square(x),8*n),
        ('mean',lambda:torch.mean(x,dim=-1),4*(n+m)),
        ('mul',lambda:x*x,8*n),
        ('copy',lambda:x.clone(),8*n)]:
      bench(f'{label}_b{batch}',fn,bytes_min=bytes_min,queued=16,
            metadata={'elements':n,'rows':m,'width':h})
    del x,norm;gc.collect();torch.mps.empty_cache()
    x=torch.randn(batch,1024,9216,device='mps');up=torch.randn_like(x);n=x.numel()
    bench(f'gelu_b{batch}',lambda:torch.nn.functional.gelu(x,approximate='tanh'),
          bytes_min=8*n,queued=8,metadata={'elements':n,'nonlinear':'tanh'})
    bench(f'gelu_mul_b{batch}',lambda:torch.nn.functional.gelu(x,approximate='tanh')*up,
          bytes_min=12*n,queued=8,metadata={'elements':n,'nonlinear':'tanh'})
    del x,up;gc.collect();torch.mps.empty_cache()

    q=torch.randn(batch,1024,8,256,device='mps').transpose(1,2)
    k=torch.randn(batch,1024,4,256,device='mps').transpose(1,2);v=torch.randn_like(k)
    triangles=1024*1025//2
    bench(f'sdpa_causal_b{batch}',lambda:torch.nn.functional.scaled_dot_product_attention(
            q,k,v,is_causal=True,enable_gqa=True,scale=256**-0.5),
          flops=4*batch*8*256*triangles,
          bytes_min=4*(2*q.numel()+k.numel()+v.numel()),queued=8,
          metadata={'B':batch,'S':1024,'Hq':8,'Hkv':4,'D':256,
                    'flops_meaning':'useful causal QK+AV only; excludes softmax and any redundant tile work',
                    'query_stride':list(q.stride()),'key_stride':list(k.stride())})
    cos=torch.randn(1,1024,256,device='mps');sin=torch.randn_like(cos)
    bench(f'rope_b{batch}',lambda:apply_rotary_pos_emb(q,k,cos,sin),
          flops=3*(q.numel()+k.numel()),
          bytes_min=4*(2*(q.numel()+k.numel())+cos.numel()+sin.numel()),queued=16,
          metadata={'B':batch,'S':1024,'Hq':8,'Hkv':4,'D':256})
    del q,k,v,cos,sin;gc.collect();torch.mps.empty_cache()

  report['status']='complete'
  report['peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
  write_json(output,report)


if __name__=='__main__':
  main()
