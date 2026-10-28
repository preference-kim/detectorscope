// Same FMA kernel body as the reference calibration, varying only work supply
// and threads per threadgroup. This does not tune MPS's private GEMM kernel.
#define main roofline_reference_main
#include "roofline_native.mm"
#undef main

int main() {
  @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device) return 1;
    id<MTLCommandQueue> queue=[device newCommandQueue];
    NSError* error=nil;
    id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithUTF8String:shader]
                                         options:nil error:&error];
    if(!lib) {fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
    printf("operation,working_set_bytes,threads,iterations,group,dispatches,sample,gpu_seconds,wall_seconds,logical_bytes,flops\n");
    // Interleave work-supply levels across group sizes. Repeat samples stay
    // within each case; this is a scheduling sensitivity probe, not autotuning.
    for(uint32_t groups:{16u,256u,64u,1024u})
      for(uint32_t group:{128u,32u,512u,64u,256u})
        run(device,queue,lib,"fp32_fma",size_t(groups)*group*4,
            groups*group,8192,group,8);
  }
}
