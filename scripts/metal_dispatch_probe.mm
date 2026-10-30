// Measurement-only public Metal API interception inside the benchmark process.
// No model/kernel changes and no hardware-core affinity or occupancy measurement.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <mutex>

static std::mutex guard;
static NSMutableArray* records;
static NSDictionary* information;
static NSString* activeLabel;
static char pipelineKey, memoryKey;
static IMP originalSetPipeline, originalGroups, originalThreads, originalMemory;

static NSArray* sizeArray(MTLSize s) { return @[@(s.width), @(s.height), @(s.depth)]; }

static void setPipeline(id self, SEL sel, id<MTLComputePipelineState> pipeline) {
  objc_setAssociatedObject(self, &pipelineKey, pipeline, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  reinterpret_cast<void(*)(id,SEL,id)>(originalSetPipeline)(self,sel,pipeline);
}

static void setMemory(id self, SEL sel, NSUInteger length, NSUInteger index) {
  NSMutableDictionary* lengths=objc_getAssociatedObject(self,&memoryKey);
  if(!lengths) {
    lengths=[NSMutableDictionary dictionary];
    objc_setAssociatedObject(self,&memoryKey,lengths,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  }
  lengths[@(index)]=@(length);
  reinterpret_cast<void(*)(id,SEL,NSUInteger,NSUInteger)>(originalMemory)(self,sel,length,index);
}

static void record(id self, NSString* kind, MTLSize grid, MTLSize group) {
  std::lock_guard<std::mutex> lock(guard);
  if(!activeLabel) return;
  id<MTLComputePipelineState> pipeline=objc_getAssociatedObject(self,&pipelineKey);
  NSMutableDictionary* lengths=objc_getAssociatedObject(self,&memoryKey);
  NSUInteger dynamicBytes=0;
  for(NSNumber* value in lengths.allValues) dynamicBytes+=value.unsignedIntegerValue;
  [records addObject:@{@"case":activeLabel,@"kind":kind,@"grid":sizeArray(grid),
    @"threads_per_threadgroup":sizeArray(group),@"pipeline_label":pipeline.label?:@"",
    @"thread_execution_width":@(pipeline.threadExecutionWidth),
    @"pipeline_max_threads":@(pipeline.maxTotalThreadsPerThreadgroup),
    @"static_threadgroup_bytes":@(pipeline.staticThreadgroupMemoryLength),
    @"dynamic_threadgroup_bytes":@(dynamicBytes)}];
}

static void groups(id self, SEL sel, MTLSize grid, MTLSize group) {
  record(self,@"threadgroups",grid,group);
  reinterpret_cast<void(*)(id,SEL,MTLSize,MTLSize)>(originalGroups)(self,sel,grid,group);
}

static void threads(id self, SEL sel, MTLSize grid, MTLSize group) {
  record(self,@"threads",grid,group);
  reinterpret_cast<void(*)(id,SEL,MTLSize,MTLSize)>(originalThreads)(self,sel,grid,group);
}

static IMP replace(Class cls, SEL sel, IMP replacement) {
  Method method=class_getInstanceMethod(cls,sel);
  IMP previous=method_getImplementation(method);
  class_addMethod(cls,sel,previous,method_getTypeEncoding(method));
  class_replaceMethod(cls,sel,replacement,method_getTypeEncoding(method));
  return previous;
}

extern "C" int probe_install() {
  @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device) return 1;
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    Class cls=object_getClass(encoder);
    records=[NSMutableArray array];
    NSMutableArray* counters=[NSMutableArray array];
    for(id<MTLCounterSet> set in device.counterSets) {
      NSMutableArray* names=[NSMutableArray array];
      for(id<MTLCounter> counter in set.counters) [names addObject:counter.name];
      [counters addObject:@{@"set":set.name,@"counters":names}];
    }
    information=@{@"device":device.name,@"encoder_class":NSStringFromClass(cls),
      @"dispatch_counter_sampling":@([device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]),
      @"stage_counter_sampling":@([device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]),
      @"counter_sets":counters,@"max_threadgroup_memory":@(device.maxThreadgroupMemoryLength)};
    originalSetPipeline=replace(cls,@selector(setComputePipelineState:),reinterpret_cast<IMP>(setPipeline));
    originalGroups=replace(cls,@selector(dispatchThreadgroups:threadsPerThreadgroup:),reinterpret_cast<IMP>(groups));
    originalThreads=replace(cls,@selector(dispatchThreads:threadsPerThreadgroup:),reinterpret_cast<IMP>(threads));
    originalMemory=replace(cls,@selector(setThreadgroupMemoryLength:atIndex:),reinterpret_cast<IMP>(setMemory));
    [encoder endEncoding];
    return 0;
  }
}

extern "C" void probe_start(const char* label) {
  std::lock_guard<std::mutex> lock(guard);
  activeLabel=[NSString stringWithUTF8String:label];
}
extern "C" void probe_stop() {
  std::lock_guard<std::mutex> lock(guard);
  activeLabel=nil;
}
extern "C" int probe_write(const char* path) {
  std::lock_guard<std::mutex> lock(guard);
  NSError* error=nil;
  NSData* data=[NSJSONSerialization dataWithJSONObject:@{@"device":information,@"dispatches":records}
                                             options:NSJSONWritingPrettyPrinted error:&error];
  return data && [data writeToFile:[NSString stringWithUTF8String:path] atomically:YES] ? 0:2;
}
