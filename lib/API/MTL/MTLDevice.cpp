#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#define IR_RUNTIME_METALCPP
#define IR_PRIVATE_IMPLEMENTATION
#include "metal_irconverter_runtime.h"

#include "API/Device.h"
#include "Support/Pipeline.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace offloadtest;

static llvm::Error toError(NS::Error *Err) {
  if (!Err)
    return llvm::Error::success();
  const std::error_code EC =
      std::error_code(static_cast<int>(Err->code()), std::system_category());
  llvm::SmallString<256> ErrMsg;
  llvm::raw_svector_ostream OS(ErrMsg);
  OS << Err->localizedDescription()->utf8String();
  if (Err->localizedFailureReason())
    OS << ": " << Err->localizedFailureReason()->utf8String();
  return llvm::createStringError(EC, ErrMsg);
}

#define MTLFormats(FMT)                                                        \
  if (Channels == 1)                                                           \
    return MTL::PixelFormatR##FMT;                                             \
  if (Channels == 2)                                                           \
    return MTL::PixelFormatRG##FMT;                                            \
  if (Channels == 4)                                                           \
    return MTL::PixelFormatRGBA##FMT;

static MTL::PixelFormat getMTLFormat(DataFormat Format, int Channels) {
  switch (Format) {
  case DataFormat::Int32:
    MTLFormats(32Sint) break;
  case DataFormat::Float32:
    MTLFormats(32Float) break;
  default:
    llvm_unreachable("Unsupported Resource format specified");
  }
  return MTL::PixelFormatInvalid;
}

#define MTLVTXFormats(Base)                                                    \
  if (Channels == 1)                                                           \
    return MTL::VertexFormat##Base;                                            \
  if (Channels == 2)                                                           \
    return MTL::VertexFormat##Base##2;                                         \
  if (Channels == 3)                                                           \
    return MTL::VertexFormat##Base##3;                                         \
  if (Channels == 4)                                                           \
    return MTL::VertexFormat##Base##4;

static MTL::VertexFormat getMTLVertexFormat(DataFormat Format, int Channels) {
  switch (Format) {
  case DataFormat::Float32:
    MTLVTXFormats(Float) break;
  default:
    llvm_unreachable("Unsupported Resource format specified");
  }
  return MTL::VertexFormatInvalid;
}

namespace {
class MTLDevice : public offloadtest::Device {
  Capabilities Caps;
  MTL::Device *Device;

  struct InvocationState {
    InvocationState() { Pool = NS::AutoreleasePool::alloc()->init(); }
    ~InvocationState() {
      for (MTL::Texture *T : Textures)
        T->release();
      for (MTL::Buffer *B : Buffers)
        B->release();
      if (ComputePipeline)
        ComputePipeline->release();
      if (RenderPipeline)
        RenderPipeline->release();
      if (Queue)
        Queue->release();

      Pool->release();
    }

    NS::AutoreleasePool *Pool = nullptr;
    MTL::CommandQueue *Queue = nullptr;
    MTL::ComputePipelineState *ComputePipeline = nullptr;
    MTL::RenderPipelineState *RenderPipeline = nullptr;
    MTL::Buffer *ArgBuffer;
    MTL::Buffer *VertexBuffer;
    MTL::VertexDescriptor *VertexDescriptor;
    llvm::SmallVector<MTL::Texture *> Textures;
    llvm::SmallVector<MTL::Buffer *> Buffers;
    MTL::Texture *FrameBufferTexture = nullptr;
  };

  llvm::Error setupVertexShader(InvocationState &IS, const Pipeline &P) {
    if (P.Bindings.VertexBufferPtr) {
      const size_t ResourceCount = P.getDescriptorCount();
      IS.VertexDescriptor = MTL::VertexDescriptor::alloc()->init();
      const uint32_t Stride = P.Bindings.getVertexStride();
      for (size_t I = 0; I < P.Bindings.VertexAttributes.size(); ++I) {
        const VertexAttribute &VA = P.Bindings.VertexAttributes[I];
        MTL::VertexAttributeDescriptor *VADesc =
            MTL::VertexAttributeDescriptor::alloc()->init();
        VADesc->setBufferIndex(ResourceCount);
        VADesc->setOffset(VA.Offset);
        VADesc->setFormat(getMTLVertexFormat(VA.Format, VA.Channels));
        IS.VertexDescriptor->attributes()->setObject(VADesc, I);
      }

      MTL::VertexBufferLayoutDescriptor *LDesc =
          MTL::VertexBufferLayoutDescriptor::alloc()->init();
      LDesc->setStride(Stride);
      LDesc->setStepRate(1);
      LDesc->setStepFunction(MTL::VertexStepFunctionPerVertex);
      IS.VertexDescriptor->layouts()->setObject(LDesc, 0);
    }
    return llvm::Error::success();
  }

  llvm::Error loadShaders(InvocationState &IS, const Pipeline &P) {
    NS::Error *Error = nullptr;
    if (P.isCompute()) {
      const llvm::StringRef Program = P.Shaders[0].Shader->getBuffer();
      dispatch_data_t Data = dispatch_data_create(
          Program.data(), Program.size(), dispatch_get_main_queue(),
          ^{
          });
      MTL::Library *Lib = Device->newLibrary(Data, &Error);
      if (Error)
        return toError(Error);
      IS.Pool->addObject(Lib);

      MTL::Function *Fn = Lib->newFunction(NS::String::string(
          P.Shaders[0].Entry.c_str(), NS::UTF8StringEncoding));
      IS.ComputePipeline = Device->newComputePipelineState(Fn, &Error);
      if (Error)
        return toError(Error);
      IS.Pool->addObject(Fn);
    } else {
      MTL::RenderPipelineDescriptor *Desc =
          MTL::RenderPipelineDescriptor::alloc()->init();
      IS.Pool->addObject(Desc);
      for (const auto &S : P.Shaders) {
        const llvm::StringRef Program = S.Shader->getBuffer();
        dispatch_data_t Data = dispatch_data_create(
            Program.data(), Program.size(), dispatch_get_main_queue(),
            ^{
            });
        MTL::Library *Lib = Device->newLibrary(Data, &Error);
        if (Error)
          return toError(Error);
        IS.Pool->addObject(Lib);

        MTL::Function *Fn = Lib->newFunction(
            NS::String::string(S.Entry.c_str(), NS::UTF8StringEncoding));
        switch (S.Stage) {
        case Stages::Vertex:
          Desc->setVertexFunction(Fn);
          if (llvm::Error Err = setupVertexShader(IS, P))
            return Err;
          Desc->setVertexDescriptor(IS.VertexDescriptor);
          break;
        case Stages::Pixel:
          Desc->setFragmentFunction(Fn);
          break;
        case Stages::Compute:
          return llvm::createStringError(
              std::errc::not_supported,
              "Metal: Compute shader invalid with render pipeline!");
        }
        if (Error)
          return toError(Error);
        IS.Pool->addObject(Fn);
      }

      // Make sure the pipeline color attachment format matches the intended
      // render target so the pipeline is compiled with the correct format.
      if (P.Bindings.RTargetBufferPtr) {
        const MTL::PixelFormat PF =
            getMTLFormat(P.Bindings.RTargetBufferPtr->Format,
                         P.Bindings.RTargetBufferPtr->Channels);
        auto *CADesc = Desc->colorAttachments()->object(0);
        if (CADesc)
          CADesc->setPixelFormat(PF);
      }

      IS.RenderPipeline = Device->newRenderPipelineState(Desc, &Error);
      if (Error)
        return toError(Error);
    }

    return llvm::Error::success();
  }

  llvm::Error createDescriptor(Resource &R, InvocationState &IS,
                               const uint32_t HeapIdx) {
    auto *TablePtr = (IRDescriptorTableEntry *)IS.ArgBuffer->contents();

    assert(R.BufferPtr->ArraySize == 1 &&
           "Resource arrays are not yet supported on Metal.");

    if (R.isRaw()) {
      MTL::Buffer *Buf =
          Device->newBuffer(R.BufferPtr->Data.back().get(), R.size(),
                            MTL::ResourceStorageModeManaged);
      IRBufferView View = {};
      View.buffer = Buf;
      View.bufferSize = R.size();

      IRDescriptorTableSetBufferView(&TablePtr[HeapIdx], &View);
      IS.Buffers.push_back(Buf);
    } else {
      const uint64_t Width = R.isTexture() ? R.BufferPtr->OutputProps.Width
                                           : R.size() / R.getElementSize();
      const uint64_t Height =
          R.isTexture() ? R.BufferPtr->OutputProps.Height : 1;
      MTL::TextureUsage UsageFlags = MTL::ResourceUsageRead;
      if (R.isReadWrite())
        UsageFlags |= MTL::ResourceUsageWrite;
      MTL::TextureDescriptor *Desc = nullptr;
      const MTL::PixelFormat Format =
          getMTLFormat(R.BufferPtr->Format, R.BufferPtr->Channels);
      switch (R.Kind) {
      case ResourceKind::Buffer:
      case ResourceKind::RWBuffer:
        Desc = MTL::TextureDescriptor::textureBufferDescriptor(
            Format, Width, MTL::ResourceStorageModeManaged, UsageFlags);
        break;
      case ResourceKind::Texture2D:
      case ResourceKind::RWTexture2D:
        Desc = MTL::TextureDescriptor::texture2DDescriptor(Format, Width,
                                                           Height, false);
        break;
      case ResourceKind::StructuredBuffer:
      case ResourceKind::RWStructuredBuffer:
      case ResourceKind::ByteAddressBuffer:
      case ResourceKind::RWByteAddressBuffer:
      case ResourceKind::ConstantBuffer:
        llvm_unreachable("Raw is checked above");
      }

      MTL::Texture *NewTex = Device->newTexture(Desc);
      NewTex->replaceRegion(MTL::Region(0, 0, Width, Height), 0,
                            R.BufferPtr->Data.back().get(),
                            Width * R.getElementSize());

      IS.Textures.push_back(NewTex);

      IRDescriptorTableSetTexture(&TablePtr[HeapIdx], NewTex, 0, 0);
    }

    return llvm::Error::success();
  }

  llvm::Error createBuffers(Pipeline &P, InvocationState &IS) {
    const size_t GraphicsDescriptorCount = P.isGraphics() ? 1 : 0;
    const size_t ResourceCount = P.getDescriptorCount();
    const size_t TableSize = sizeof(IRDescriptorTableEntry) *
                             (ResourceCount + GraphicsDescriptorCount);
    IS.ArgBuffer =
        Device->newBuffer(TableSize, MTL::ResourceStorageModeManaged);

    uint32_t HeapIndex = 0;
    for (auto &D : P.Sets) {
      for (auto &R : D.Resources) {
        if (auto Err = createDescriptor(R, IS, HeapIndex++))
          return Err;
      }
    }
    if (P.isGraphics()) {
      IS.VertexBuffer = Device->newBuffer(
          P.Bindings.VertexBufferPtr->Data.back().get(),
          P.Bindings.VertexBufferPtr->size(), MTL::ResourceStorageModeManaged);
      // Ensure GPU can see the CPU-initialized vertex data for managed buffers
      IS.VertexBuffer->didModifyRange(
          NS::Range::Make(0, IS.VertexBuffer->length()));

      // Place the vertex buffer in the argument table after all other
      // resources.
      auto *TablePtr = (IRDescriptorTableEntry *)IS.ArgBuffer->contents();
      IRBufferView View = {};
      View.buffer = IS.VertexBuffer;
      View.bufferSize = P.Bindings.VertexBufferPtr->size();
      IRDescriptorTableSetBufferView(&TablePtr[ResourceCount], &View);
    }
    IS.ArgBuffer->didModifyRange(NS::Range::Make(0, IS.ArgBuffer->length()));
    return llvm::Error::success();
  }

  llvm::Error executeCommands(Pipeline &P, InvocationState &IS) {
    MTL::CommandBuffer *CmdBuffer = IS.Queue->commandBuffer();

    if (IS.ComputePipeline) {
      MTL::ComputeCommandEncoder *CmdEncoder =
          CmdBuffer->computeCommandEncoder();

      CmdEncoder->setComputePipelineState(IS.ComputePipeline);
      CmdEncoder->setBuffer(IS.ArgBuffer, 0, 2);
      for (uint64_t I = 0; I < IS.Textures.size(); ++I)
        CmdEncoder->useResource(IS.Textures[I], MTL::ResourceUsageRead |
                                                    MTL::ResourceUsageWrite);
      for (uint64_t I = 0; I < IS.Buffers.size(); ++I)
        CmdEncoder->useResource(IS.Buffers[I], MTL::ResourceUsageRead |
                                                   MTL::ResourceUsageWrite);

      const NS::UInteger TGS =
          IS.ComputePipeline->maxTotalThreadsPerThreadgroup();
      const llvm::ArrayRef<int> DispatchSize =
          llvm::ArrayRef<int>(P.Shaders[0].DispatchSize);
      const MTL::Size GridSize =
          MTL::Size(TGS * DispatchSize[0], DispatchSize[1], DispatchSize[2]);
      const MTL::Size GroupSize(TGS, 1, 1);
      CmdEncoder->dispatchThreads(GridSize, GroupSize);
      CmdEncoder->memoryBarrier(MTL::BarrierScopeBuffers);

      CmdEncoder->endEncoding();
    } else {
      assert(IS.RenderPipeline && "If not compute... render!");
      MTL::RenderPassDescriptor *Desc =
          MTL::RenderPassDescriptor::alloc()->init();

      // Setup the render target texture.
      Buffer *RTarget = P.Bindings.RTargetBufferPtr;

      const MTL::PixelFormat Format =
          getMTLFormat(RTarget->Format, RTarget->Channels);

      const uint64_t Width = RTarget->OutputProps.Width;
      const uint64_t Height = RTarget->OutputProps.Height;
      MTL::TextureDescriptor *TDesc =
          MTL::TextureDescriptor::texture2DDescriptor(Format, Width, Height,
                                                      false);
      // Make this texture usable as a render target and shader resource. Use
      // Shared storage so the CPU can read it after command buffer completion
      // without an explicit blit/synchronization step on macOS.
      TDesc->setUsage(MTL::TextureUsageRenderTarget |
                      MTL::TextureUsageShaderRead |
                      MTL::TextureUsageShaderWrite);
      TDesc->setStorageMode(MTL::StorageModeShared);

      IS.FrameBufferTexture = Device->newTexture(TDesc);
      auto *CADesc = MTL::RenderPassColorAttachmentDescriptor::alloc()->init();
      CADesc->setTexture(IS.FrameBufferTexture);
      CADesc->setLoadAction(MTL::LoadActionClear);
      CADesc->setClearColor(MTL::ClearColor());
      CADesc->setStoreAction(MTL::StoreActionStore);
      Desc->colorAttachments()->setObject(CADesc, 0);

      MTL::RenderCommandEncoder *CmdEncoder =
          CmdBuffer->renderCommandEncoder(Desc);

      CmdEncoder->setRenderPipelineState(IS.RenderPipeline);
      CmdEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0),
                                 P.Bindings.getVertexCount());

      CmdEncoder->memoryBarrier(MTL::BarrierScopeBuffers,
                                MTL::RenderStageFragment, 0);
      CmdEncoder->endEncoding();
    }

    CmdBuffer->commit();
    CmdBuffer->waitUntilCompleted();

    return llvm::Error::success();
  }

  llvm::Error copyBack(Pipeline &P, InvocationState &IS) {
    uint32_t TextureIndex = 0;
    uint32_t BufferIndex = 0;
    for (auto &D : P.Sets) {
      for (auto &R : D.Resources) {
        assert(R.BufferPtr->ArraySize == 1 &&
               "Resource arrays are not yet supported on Metal.");
        if (R.isReadOnly()) {
          if (R.isRaw())
            ++BufferIndex;
          else
            ++TextureIndex;
          continue;
        }
        if (R.isRaw()) {
          memcpy(R.BufferPtr->Data.back().get(),
                 IS.Buffers[BufferIndex++]->contents(), R.size());
          continue;
        }
        const uint64_t Width = R.isTexture() ? R.BufferPtr->OutputProps.Width
                                             : R.size() / R.getElementSize();
        const uint64_t Height =
            R.isTexture() ? R.BufferPtr->OutputProps.Height : 1;
        IS.Textures[TextureIndex++]->getBytes(
            R.BufferPtr->Data.back().get(), Width * R.getElementSize(),
            MTL::Region(0, 0, Width, Height), 0);
      }
    }
    if (P.isGraphics()) {
      Buffer *RTarget = P.Bindings.RTargetBufferPtr;
      const uint64_t Width = RTarget->OutputProps.Width;
      const uint64_t Height = RTarget->OutputProps.Height;
      IS.FrameBufferTexture->getBytes(RTarget->Data[0].get(),
                                      Width * RTarget->getElementSize(),
                                      MTL::Region(0, 0, Width, Height), 0);
    }
    return llvm::Error::success();
  }

public:
  MTLDevice(MTL::Device *D) : Device(D) {
    Description = Device->name()->utf8String();
  }
  const Capabilities &getCapabilities() override {
    if (Caps.empty())
      queryCapabilities();
    return Caps;
  }

  llvm::StringRef getAPIName() const override { return "Metal"; };
  GPUAPI getAPI() const override { return GPUAPI::Metal; };

  llvm::Error executeProgram(Pipeline &P) override {
    InvocationState IS;
    IS.Queue = Device->newCommandQueue();

    if (auto Err = createBuffers(P, IS))
      return Err;

    if (auto Err = loadShaders(IS, P))
      return Err;

    if (auto Err = executeCommands(P, IS))
      return Err;

    if (auto Err = copyBack(P, IS))
      return Err;
    return llvm::Error::success();
  }

  virtual ~MTLDevice() {};

private:
  void queryCapabilities() {}
};

class MTLContext {
  MTLContext() = default;
  ~MTLContext() {}
  MTLContext(const MTLContext &) = delete;

  llvm::SmallVector<std::shared_ptr<MTLDevice>> Devices;

public:
  static MTLContext &instance() {
    static MTLContext Ctx;
    return Ctx;
  }

  llvm::Error initialize() {
    auto DefaultDev =
        std::make_shared<MTLDevice>(MTL::CreateSystemDefaultDevice());
    Devices.push_back(DefaultDev);
    Device::registerDevice(std::static_pointer_cast<Device>(DefaultDev));
    return llvm::Error::success();
  }
};

} // namespace

llvm::Error Device::initializeMtlDevices(const DeviceConfig /*Config*/) {
  return MTLContext::instance().initialize();
}
