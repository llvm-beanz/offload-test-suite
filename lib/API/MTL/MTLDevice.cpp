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
#include <algorithm>

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

  llvm::Error setupVertexShader(InvocationState &IS, const Pipeline &P,
                                MTL::Function *Fn) {
    if (P.Bindings.VertexBufferPtr) {
      NS::Array *FnAttrs = Fn->vertexAttributes();
      // I'm not really sure if there's any valid case for a vertex shader with
      // no vertex attributes, so we just error if that ever occurs.
      if (!FnAttrs)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Vertex shader has no vertex attributes.");
      if (FnAttrs->count() != P.Bindings.VertexAttributes.size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Mismatch between vertex shader attribute count and pipeline "
            "vertex input count.");
      // Collect the attribute indices the shader expects so that we can map the
      // specified attributes onto the correct indices.
      llvm::SmallVector<uint32_t> ShaderAttrIndices;
      for (uint32_t Ai = 0; Ai < FnAttrs->count(); ++Ai) {
        auto *A = static_cast<MTL::VertexAttribute *>(FnAttrs->object(Ai));
        if (A && A->active())
          ShaderAttrIndices.push_back(A->attributeIndex());
      }

      IS.VertexDescriptor = MTL::VertexDescriptor::alloc()->init();
      const uint32_t Stride = P.Bindings.getVertexStride();
      for (size_t I = 0; I < P.Bindings.VertexAttributes.size(); ++I) {
        const VertexAttribute &VA = P.Bindings.VertexAttributes[I];
        MTL::VertexAttributeDescriptor *VADesc =
            MTL::VertexAttributeDescriptor::alloc()->init();
        // Vertex attributes are sourced from the vertex buffer bound to the
        // vertex buffer index 0 on the encoder (see executeCommands()). Use
        // buffer index 0 here so the descriptor references the same slot.
        VADesc->setBufferIndex(0);
        VADesc->setOffset(VA.Offset);
        VADesc->setFormat(getMTLVertexFormat(VA.Format, VA.Channels));
        IS.VertexDescriptor->attributes()->setObject(VADesc,
                                                     ShaderAttrIndices[I]);
      }

      MTL::VertexBufferLayoutDescriptor *LDesc =
          MTL::VertexBufferLayoutDescriptor::alloc()->init();
      LDesc->setStride(Stride);
      LDesc->setStepRate(1);
      LDesc->setStepFunction(MTL::VertexStepFunctionPerVertex);
      IS.VertexDescriptor->layouts()->setObject(LDesc, 0);

      // Debug: print shader attribute indices and final vertex descriptor
      // mapping.
      llvm::errs() << "Shader attribute indices: ";
      for (auto Idx : ShaderAttrIndices)
        llvm::errs() << Idx << " ";
      llvm::errs() << "\n";
      llvm::errs() << "Vertex descriptor attributes present at indices:";
      for (uint32_t Ai = 0; Ai < 31; ++Ai) {
        auto *AD = static_cast<MTL::VertexAttributeDescriptor *>(
            IS.VertexDescriptor->attributes()->object(Ai));
        if (AD)
          llvm::errs() << " " << Ai;
      }
      llvm::errs() << "\n";
      // Print details for the attributes the shader expects.
      for (uint32_t Idx = 0; Idx < ShaderAttrIndices.size(); ++Idx) {
        const uint32_t AttrIndex = ShaderAttrIndices[Idx];
        auto *AD = static_cast<MTL::VertexAttributeDescriptor *>(
            IS.VertexDescriptor->attributes()->object(AttrIndex));
        if (AD) {
          llvm::errs() << "Attribute[" << AttrIndex
                       << "] bufferIndex=" << AD->bufferIndex()
                       << " offset=" << AD->offset()
                       << " format=" << (int)AD->format() << "\n";
        }
      }
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
          if (llvm::Error Err = setupVertexShader(IS, P, Fn))
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
        // Ensure a color attachment descriptor exists on the pipeline
        // descriptor and set its pixel format to match the render target.
        MTL::RenderPipelineColorAttachmentDescriptor *RPCA =
            MTL::RenderPipelineColorAttachmentDescriptor::alloc()->init();
        RPCA->setPixelFormat(PF);
        Desc->colorAttachments()->setObject(RPCA, 0);
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
    const size_t ResourceCount = P.getDescriptorCount();
    const size_t TableSize = sizeof(IRDescriptorTableEntry) * ResourceCount;

    if (TableSize > 0) {
      IS.ArgBuffer =
          Device->newBuffer(TableSize, MTL::ResourceStorageModeManaged);
      uint32_t HeapIndex = 0;
      for (auto &D : P.Sets) {
        for (auto &R : D.Resources) {
          if (auto Err = createDescriptor(R, IS, HeapIndex++))
            return Err;
        }
      }
      IS.ArgBuffer->didModifyRange(NS::Range::Make(0, IS.ArgBuffer->length()));
    }
    if (P.isGraphics()) {
      IS.VertexBuffer = Device->newBuffer(
          P.Bindings.VertexBufferPtr->Data.back().get(),
          P.Bindings.VertexBufferPtr->size(), MTL::ResourceStorageModeManaged);
      // Ensure GPU can see the CPU-initialized vertex data for managed buffers
      IS.VertexBuffer->didModifyRange(
          NS::Range::Make(0, IS.VertexBuffer->length()));
    }
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
      // Create a single shared texture used for both rendering and CPU
      // readback. Rendering directly into a shared texture can be less
      // efficient on some drivers, but this simplifies the path and lets us
      // read back without an explicit blit.
      MTL::TextureDescriptor *SharedDesc = TDesc->copy();
      SharedDesc->setUsage(MTL::TextureUsageRenderTarget |
                           MTL::TextureUsageShaderRead |
                           MTL::TextureUsageShaderWrite);
      SharedDesc->setStorageMode(MTL::StorageModeShared);
      IS.FrameBufferTexture = Device->newTexture(SharedDesc);

      // Debug: print texture properties so we can verify formats and storage
      // modes used for render and readback.
      if (IS.FrameBufferTexture) {
        llvm::errs() << "FrameBufferTexture: fmt="
                     << (int)IS.FrameBufferTexture->pixelFormat()
                     << " storageMode="
                     << (int)IS.FrameBufferTexture->storageMode()
                     << " width=" << IS.FrameBufferTexture->width()
                     << " height=" << IS.FrameBufferTexture->height() << "\n";
      }

      auto *CADesc = MTL::RenderPassColorAttachmentDescriptor::alloc()->init();
      CADesc->setTexture(IS.FrameBufferTexture);
      CADesc->setLoadAction(MTL::LoadActionClear);
      // Revert diagnostic clear to default (black). We previously cleared to
      // red to verify the blit/readback path; that diagnostic is no longer
      // needed and would obscure the actual fragment shader output.
      CADesc->setClearColor(MTL::ClearColor());
      CADesc->setStoreAction(MTL::StoreActionStore);
      Desc->colorAttachments()->setObject(CADesc, 0);

      MTL::RenderCommandEncoder *CmdEncoder =
          CmdBuffer->renderCommandEncoder(Desc);

      CmdEncoder->setRenderPipelineState(IS.RenderPipeline);
      // Explicitly set viewport to texture dimensions to avoid relying on any
      // default behavior that might differ across drivers.
      CmdEncoder->setViewport(
          MTL::Viewport{0.0, 0.0, (double)Width, (double)Height, 0.0, 1.0});
      // Disable face culling for diagnostics; some shaders/vertex orders
      // may produce culled triangles depending on winding conventions.
      CmdEncoder->setCullMode(MTL::CullModeNone);
      // Bind vertex buffer at slot 0 to match the vertex descriptor which
      // references buffer index 0.
      CmdEncoder->setVertexBuffer(IS.VertexBuffer, 0, 0);
      // Debug: print vertex count and verify vertex buffer length.
      llvm::errs() << "Drawing vertices: " << P.Bindings.getVertexCount()
                   << "\n";
      llvm::errs() << "Vertex stride: " << P.Bindings.getVertexStride() << "\n";
      if (IS.VertexBuffer)
        llvm::errs() << "VertexBuffer length: " << IS.VertexBuffer->length()
                     << "\n";
      // Dump the first vertex bytes (interpreting as floats) to confirm data
      if (IS.VertexBuffer && IS.VertexBuffer->length() >= 16) {
        float *V = reinterpret_cast<float *>(IS.VertexBuffer->contents());
        llvm::errs() << "First vertex floats: ";
        for (int I = 0; I < 7; ++I)
          llvm::errs() << V[I] << " ";
        llvm::errs() << "\n";
      }
      CmdEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0),
                                 P.Bindings.getVertexCount());

      /*CmdEncoder->memoryBarrier(MTL::BarrierScopeBuffers,
                                MTL::RenderStageFragment, 0);*/
      CmdEncoder->endEncoding();

      // No blit required when rendering directly into the shared texture.
      llvm::errs()
          << "Rendering directly into shared render/readback texture\n";
    }

    CmdBuffer->commit();
    CmdBuffer->waitUntilCompleted();

    // Debug: print command buffer completion status and any reported error
    auto Status = CmdBuffer->status();
    llvm::errs() << "CmdBuffer status=" << (int)Status << "\n";
    NS::Error *CBErr = CmdBuffer->error();
    if (CBErr)
      llvm::errs() << "CmdBuffer error: "
                   << CBErr->localizedDescription()->utf8String() << "\n";

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
      const size_t ElemSize = RTarget->getElementSize();
      const size_t RowBytes = Width * ElemSize;

      // Read the framebuffer one row at a time into the output buffer.
      // Read rows from the texture bottom-to-top into the buffer top-to-bottom
      // so the final image is upright without needing a post-read flip.
      unsigned char *Buf =
          reinterpret_cast<unsigned char *>(RTarget->Data[0].get());
      for (uint64_t R = 0; R < Height; ++R) {
        const uint32_t SrcRow = (uint32_t)((Height - 1) - R);
        unsigned char *Dst = Buf + R * RowBytes;
        IS.FrameBufferTexture->getBytes(
            Dst, RowBytes, MTL::Region(0, SrcRow, (uint32_t)Width, 1), 0);
      }
      llvm::errs() << "copyBack: read rows one-at-a-time (height=" << Height
                   << ")\n";

      // Debug: dump first bytes and, if float format, the first few floats so
      // we can tell whether the readback contains any non-zero data.
      llvm::errs() << "copyBack: framebuffer readback width=" << Width
                   << " height=" << Height << " elemSize=" << ElemSize << "\n";
      unsigned char *Bytes =
          reinterpret_cast<unsigned char *>(RTarget->Data[0].get());
      const size_t DumpBytes = std::min<size_t>(RowBytes, 64);
      llvm::errs() << "First " << DumpBytes << " bytes: ";
      for (size_t I = 0; I < DumpBytes; ++I)
        llvm::errs() << llvm::format_hex_no_prefix((uint32_t)Bytes[I], 2)
                     << " ";
      llvm::errs() << "\n";
      if (ElemSize >= 4) {
        float *F = reinterpret_cast<float *>(Bytes);
        const int FDump = std::min<int>(8, (int)(RowBytes / 4));
        llvm::errs() << "First floats: ";
        for (int I = 0; I < FDump; ++I)
          llvm::errs() << F[I] << " ";
        llvm::errs() << "\n";
      }
      // Sample a pixel from row 32 (or the last row if the texture is
      // smaller) at center X to get a better indicator of rendered content
      // (avoid the cleared first rows).
      if (Height > 0) {
        const uint64_t RowIndex = (Height > 32) ? 32 : (Height - 1);
        const size_t RowOffset = RowIndex * RowBytes;
        const size_t CenterX = Width / 2;
        const size_t PixelOffset = RowOffset + CenterX * ElemSize;
        const size_t MaxDump = (RowOffset + RowBytes) - (CenterX * ElemSize);
        const size_t DumpBytes2 = std::min<size_t>(ElemSize * 4, MaxDump);
        llvm::errs() << "Row32 sample (row=" << RowIndex
                     << ") center pixel offset=" << PixelOffset
                     << " dumpBytes=" << DumpBytes2 << " bytes: ";
        for (size_t I = 0; I < DumpBytes2; ++I)
          llvm::errs() << llvm::format_hex_no_prefix(
                              (uint32_t)Bytes[PixelOffset + I], 2)
                       << " ";
        llvm::errs() << "\n";
        if (ElemSize >= 4) {
          float *F2 = reinterpret_cast<float *>(Bytes + PixelOffset);
          const int FDump2 = std::min<int>(4, (int)(DumpBytes2 / 4));
          llvm::errs() << "Row32 center floats: ";
          for (int I = 0; I < FDump2; ++I)
            llvm::errs() << F2[I] << " ";
          llvm::errs() << "\n";
        }
      }
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
