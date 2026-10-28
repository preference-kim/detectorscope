// Calibration only: no model kernels or weights. GPU command-buffer timestamps
// measure batches of dispatches; bytes mean logical traffic, not DRAM counters.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct Params { uint32_t n, repeats, span, grid; };
static const char* shader = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Params { uint n, repeats, span, grid; };
kernel void copy_buffer(device const float4* a [[buffer(0)]],
                        device float4* c [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  c[i] = a[i];
}
kernel void triad(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                  device float4* c [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  c[i] = a[i] + 0.5f * b[i];
}
kernel void read_reuse(device const volatile float4* a [[buffer(0)]],
                       device float4* c [[buffer(2)]], constant Params& p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
  float4 acc = 0.0f;
  for (uint r = 0; r < p.repeats; ++r) acc += a[(i + r * p.grid) & (p.span - 1)];
  c[i] = acc;
}
kernel void shared_reuse(device const float4* a [[buffer(0)]],
                         device float4* c [[buffer(2)]], constant Params& p [[buffer(3)]],
                         uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup volatile float4 shared[1024];
  for (uint j = t; j < 1024; j += 256) shared[j] = a[j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 acc = 0.0f;
  for (uint r = 0; r < p.repeats; ++r) acc += shared[(t + r * 256) & 1023];
  c[i] = acc;
}
kernel void fp32_fma(device const float* a [[buffer(0)]],
                     device float* c [[buffer(2)]], constant Params& p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  float v = a[i];
  float x0=v, x1=v+0.1f, x2=v+0.2f, x3=v+0.3f;
  float x4=v+0.4f, x5=v+0.5f, x6=v+0.6f, x7=v+0.7f;
  for (uint r = 0; r < p.repeats; ++r) {
    x0=fma(x0,0.9991f,0.0001f); x1=fma(x1,0.9992f,0.0002f);
    x2=fma(x2,0.9993f,0.0003f); x3=fma(x3,0.9994f,0.0004f);
    x4=fma(x4,0.9995f,0.0005f); x5=fma(x5,0.9996f,0.0006f);
    x6=fma(x6,0.9997f,0.0007f); x7=fma(x7,0.9998f,0.0008f);
  }
  c[i] = x0+x1+x2+x3+x4+x5+x6+x7;
}
)MSL";

static void run(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> lib,
                const char* name, size_t span_bytes, uint32_t grid,
                uint32_t repeats, uint32_t group, int dispatches) {
  @autoreleasepool {
    NSError* error = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipe) { fprintf(stderr, "%s\n", error.description.UTF8String); exit(2); }
    size_t output_bytes = std::max(span_bytes, size_t(grid)*16);
    id<MTLBuffer> a = [device newBufferWithLength:span_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [device newBufferWithLength:span_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    if (!a || !b || !c) { fprintf(stderr,"bounded allocation failed\n"); exit(2); }
    for (size_t i=0; i<span_bytes/4; ++i) {
      static_cast<float*>(a.contents)[i]=0.125f;
      static_cast<float*>(b.contents)[i]=0.25f;
    }
    Params p{uint32_t(span_bytes/16),repeats,uint32_t(span_bytes/16),grid};
    for (int sample=-2; sample<7; ++sample) {
      @autoreleasepool {
        auto started=std::chrono::steady_clock::now();
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:a offset:0 atIndex:0]; [enc setBuffer:b offset:0 atIndex:1];
        [enc setBuffer:c offset:0 atIndex:2]; [enc setBytes:&p length:sizeof(p) atIndex:3];
        for(int j=0;j<dispatches;++j)
          [enc dispatchThreads:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
        [enc endEncoding]; [command commit]; [command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted) {
          fprintf(stderr,"%s\n", command.error.description.UTF8String); exit(3);
        }
        double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()/dispatches;
        double gpu=(command.GPUEndTime-command.GPUStartTime)/dispatches;
        float observed=static_cast<float*>(c.contents)[0];
        float expected=0;
        if(!strcmp(name,"copy_buffer")) expected=0.125f;
        else if(!strcmp(name,"triad")) expected=0.25f;
        else if(!strcmp(name,"read_reuse") || !strcmp(name,"shared_reuse")) expected=repeats*0.125f;
        else {
          float x[8]; for(int j=0;j<8;++j)x[j]=0.125f+0.1f*j;
          for(uint32_t rr=0;rr<repeats;++rr) for(int j=0;j<8;++j)
            x[j]=std::fma(x[j],0.9991f+0.0001f*j,0.0001f*(j+1));
          for(float xx:x)expected+=xx;
        }
        if(!std::isfinite(observed) || std::abs(observed-expected)>0.002f*std::max(1.0f,std::abs(expected))) {
          fprintf(stderr,"validation failed: %s got=%g expected=%g\n",name,observed,expected); exit(4);
        }
        double bytes=0, flops=0;
        if(!strcmp(name,"copy_buffer")) bytes=2.0*span_bytes;
        else if(!strcmp(name,"triad")) {bytes=3.0*span_bytes;flops=2.0*span_bytes/4;}
        else if(!strcmp(name,"read_reuse") || !strcmp(name,"shared_reuse")) {
          bytes=double(grid)*repeats*16; flops=double(grid)*repeats*4;
        } else {flops=double(grid)*repeats*16; bytes=double(grid)*8;}
        if(sample>=0) printf("%s,%zu,%u,%u,%u,%d,%d,%.9f,%.9f,%.0f,%.0f\n",
            name,span_bytes,grid,repeats,group,dispatches,sample,gpu,wall,bytes,flops);
      }
    }
  }
}

int main() {
  @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device){fprintf(stderr,"No Metal device\n");return 1;}
    fprintf(stderr,"device=%s unified=%d maxBufferLength=%lu\n",device.name.UTF8String,device.hasUnifiedMemory,(unsigned long)device.maxBufferLength);
    id<MTLCommandQueue> queue=[device newCommandQueue];
    NSError* error=nil;
    id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithUTF8String:shader] options:nil error:&error];
    if(!lib){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
    printf("operation,working_set_bytes,threads,iterations,group,dispatches,sample,gpu_seconds,wall_seconds,logical_bytes,flops\n");
    for(uint32_t group:{128u,256u,512u})
      run(device,queue,lib,"fp32_fma",65536*4,65536,8192,group,8);
    for(size_t bytes:{4096ul,32768ul,262144ul,1048576ul,4194304ul,16777216ul,67108864ul,134217728ul}) {
      run(device,queue,lib,"copy_buffer",bytes,uint32_t(bytes/16),1,256,8);
      run(device,queue,lib,"read_reuse",bytes,65536,128,256,8);
    }
    run(device,queue,lib,"triad",134217728,134217728/16,1,256,8);
    run(device,queue,lib,"shared_reuse",16384,65536,128,256,8);
  }
}
