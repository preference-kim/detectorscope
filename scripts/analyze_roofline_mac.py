"""Analyze local calibration data without running inference or loading weights.

The figure needs matplotlib; the JSON and CSV calculations use the standard
library. Reference efficiencies are not hardware utilization counters.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--directory', type=Path, default=Path('outputs/mac-roofline'))
  parser.add_argument('--module-summary', type=Path,
                      default=Path('outputs/mac-module-profile/summary.json'))
  parser.add_argument('--plot', action='store_true')
  args = parser.parse_args()
  root = args.directory
  inputs = [root / 'native.csv', root / 'operators.json', args.module_summary]
  native_groups = defaultdict(list)
  with inputs[0].open() as file:
    for row in csv.DictReader(file):
      key = (row['operation'], int(row['working_set_bytes']), int(row['group']))
      native_groups[key].append(row)
  native = []
  for (operation, span, group), rows in native_groups.items():
    times = [float(r['gpu_seconds']) for r in rows]
    seconds = statistics.median(times)
    native.append(dict(operation=operation, input_span_bytes=span, group=group,
                       median_gpu_seconds=seconds, min_gpu_seconds=min(times),
                       max_gpu_seconds=max(times), samples=len(rows),
                       logical_bytes_per_second=float(rows[0]['logical_bytes']) / seconds,
                       flops_per_second=float(rows[0]['flops']) / seconds))
  compute = max(r['flops_per_second'] for r in native if r['operation']=='fp32_fma')
  large = [r['logical_bytes_per_second'] for r in native
           if r['operation'] in ('copy_buffer', 'triad', 'read_reuse')
           and r['input_span_bytes'] >= 64 * 2**20]
  bandwidth = max(large)
  operators = json.loads(inputs[1].read_text())
  assert operators['status']=='complete'
  rows = []
  for op in operators['runs']:
    seconds = op['queued_median_seconds']
    flops, byte_count = op['flops'], op['minimum_tensor_bytes']
    reference = max((flops or 0) / compute, byte_count / bandwidth)
    times = [r['seconds'] for r in op['samples'] if r['dispatches']==op['queued_dispatches']]
    row = dict(name=op['name'], flops=flops, minimum_tensor_bytes=byte_count,
               arithmetic_intensity=flops / byte_count if flops else None,
               single_ms=op['single_median_seconds'] * 1000,
               queued_ms=seconds * 1000, queued_min_ms=min(times) * 1000,
               queued_max_ms=max(times) * 1000,
               achieved_tflops=flops / seconds / 1e12 if flops else None,
               effective_minimum_gbs=byte_count / seconds / 1e9,
               reference_ms=reference * 1000,
               reference_efficiency_percent=reference / seconds * 100)
    # Source-level materialized-tensor traffic, not measured DRAM traffic.
    if op['name'].startswith('rmsnorm'):
      row['unfused_logical_bytes']=op['metadata']['unfused_tensor_traffic_estimate']
    elif op['name'].startswith('gelu_mul'):
      row['unfused_logical_bytes']=20 * op['metadata']['elements']
    elif op['name'].startswith('rope'):
      elements=op['metadata']['B'] * op['metadata']['S'] * (8+4) * 256
      # Per q/k: neg half (1N), cat (2N), two muls (4N), add (3N).
      # Each distinct cos/sin buffer is read for q and k (four reads).
      row['unfused_logical_bytes']=4 * (10 * elements + 4 * 1024 * 256)
    if 'unfused_logical_bytes' in row:
      row['unfused_reference_ms']=row['unfused_logical_bytes'] / bandwidth * 1000
      row['effective_unfused_gbs']=row['unfused_logical_bytes'] / seconds / 1e9
    rows.append(row)
  by_name={row['name']:row for row in rows}
  modules=json.loads(inputs[2].read_text())['cases']['pair_equal_1024']['module_totals_median_ms']
  linear_mapping={'mlp.gate_proj':'gate_up_b2', 'mlp.up_proj':'gate_up_b2',
                  'mlp.down_proj':'down_b2', 'self_attn.q_proj':'q_b2',
                  'self_attn.k_proj':'kv_b2', 'self_attn.v_proj':'kv_b2',
                  'self_attn.o_proj':'o_b2'}
  streamed=[]
  for module, name in linear_mapping.items():
    flops=23 * by_name[name]['flops']
    seconds=modules[module] / 1000
    streamed.append(dict(module=module, blocks=23, total_ms=modules[module],
                         useful_flops=flops, achieved_tflops=flops/seconds/1e12,
                         compute_reference_percent=flops/seconds/compute*100))
  report=dict(reference_meaning='Empirical sustained reference, not architectural peak or hardware utilization',
              compute_reference_tflops=compute/1e12,
              large_buffer_bandwidth_range_gbs=[min(large)/1e9, max(large)/1e9],
              ridge_flops_per_byte=compute/bandwidth, native=native, operators=rows,
              streamed_equal_b2_before_padding_change=streamed,
              input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
  fields=list(dict.fromkeys(k for row in rows for k in row))
  with (root/'operations.csv').open('w') as file:
    writer=csv.DictWriter(file, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
  if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'svg.fonttype':'none'})
    fig, (ax, memory) = plt.subplots(1, 2, figsize=(13, 5.6), layout='constrained')
    intensity=np.logspace(-1, 3.4, 300)
    ax.loglog(intensity, np.minimum(compute, intensity*bandwidth)/1e12,
              color='#56697d', linewidth=2, label='Empirical compute + large-buffer reference')
    ax.loglog(intensity, np.minimum(compute, intensity*273e9)/1e12,
              color='#9daab5', linestyle=':', label='Apple DRAM specification + same compute reference')
    projections=[r for r in rows if r['name'].endswith('_b2') and
                 r['name'].split('_')[0] in ('gate','down','q','kv','o')]
    ax.scatter([r['arithmetic_intensity'] for r in projections],
               [r['achieved_tflops'] for r in projections], color='#087f8c', s=38, zorder=4)
    ax.annotate('B2 projections: 4.61–4.82 TFLOP/s', (485,4.78),
                xytext=(-100,-26), textcoords='offset points', fontsize=9)
    labels=[('gemv_gate_up','GEMV',(10,7)),('sdpa_causal_b2','Causal SDPA',(8,-17)),
            ('rmsnorm_b2','RMSNorm',(10,8)),('rope_b2','RoPE',(-37,-17))]
    for name, label, offset in labels:
      r=by_name[name]
      ax.scatter(r['arithmetic_intensity'],r['achieved_tflops'],s=45,color='#d16a3b',zorder=4)
      ax.annotate(label,(r['arithmetic_intensity'],r['achieved_tflops']),
                  xytext=offset,textcoords='offset points',fontsize=9)
    ax.set(xlabel='Useful FLOPs / minimum tensor byte', ylabel='Useful TFLOP/s',
           title='FP32 operators: B2 × 1,024 tokens; GEMV: M=1', xlim=(.1,2500), ylim=(.005,8))
    ax.grid(True, which='major', alpha=.18)
    ax.legend(loc='lower right', fontsize=7.5)
    for operation, label, color in [('copy_buffer','Copy: logical read + write','#087f8c'),
                                     ('read_reuse','Repeated read: logical bytes','#ae5b2d')]:
      points=sorted([r for r in native if r['operation']==operation], key=lambda r:r['input_span_bytes'])
      memory.loglog([r['input_span_bytes']/2**20 for r in points],
                     [r['logical_bytes_per_second']/1e9 for r in points],
                     marker='o',color=color,label=label)
    shared=next(r for r in native if r['operation']=='shared_reuse')
    memory.scatter(16/1024,shared['logical_bytes_per_second']/1e9,marker='s',
                    color='#8065a8',label='16 KiB threadgroup repeated read',zorder=5)
    memory.axhline(273,color='#9daab5',linestyle=':',label='273 GB/s official DRAM specification')
    memory.axhspan(min(large)/1e9,max(large)/1e9,color='#087f8c',alpha=.08)
    memory.set(xlabel='Input span (MiB); copy footprint = 2 × span',
                ylabel='Effective logical GB/s', title='Bandwidth depends on access pattern and reuse')
    memory.grid(True,which='major',alpha=.18)
    memory.legend(loc='lower right',fontsize=7.5)
    fig.suptitle('M4 Pro · 16 GPU cores · measured reference, not hardware utilization',fontsize=14)
    fig.supxlabel('Operator times: queued wall-clock medians. Native calibration: GPU command-buffer timestamps.\n'
                  'SDPA excludes softmax work; bytes are logical counts, not measured traffic at a cache level.',fontsize=9)
    fig.savefig(root/'roofline.png',dpi=180)
    fig.savefig(root/'roofline.svg')
    plt.close(fig)
  print(f'Wrote {root / "analysis.json"} and {root / "operations.csv"}')


if __name__=='__main__':
  main()
