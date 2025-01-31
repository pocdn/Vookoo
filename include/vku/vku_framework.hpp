////////////////////////////////////////////////////////////////////////////////
//
// Demo framework for the Vookoo for the Vookoo high level C++ Vulkan interface.
//
// (C) Vookoo Contributors, MIT License
//
// This is an optional demo framework for the Vookoo high level C++ Vulkan interface.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef VKU_FRAMEWORK_HPP
#define VKU_FRAMEWORK_HPP

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_EXPOSE_NATIVE_WIN32
#define VKU_SURFACE "VK_KHR_win32_surface"
#pragma warning(disable : 4005)
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT
#else // X11
#define VK_USE_PLATFORM_XLIB_KHR
#define GLFW_EXPOSE_NATIVE_X11
#define VKU_SURFACE "VK_KHR_xlib_surface"
#endif

#ifndef VKU_NO_GLFW
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#endif

// Undo damage done by windows.h
#undef APIENTRY
#undef None
#undef max
#undef min

#include <array>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <thread>
#include <chrono>
#include <functional>
#include <cstddef>

#include <vulkan/vulkan.hpp>
#include <vku/vku.hpp>

namespace vku {

struct FrameworkOptions
{
	int deviceID = 0;
	bool useCompute = false;
	bool useTessellationShader = false;
	bool useGeometryShader = false;
	bool useMultiView = false;
};

/// This class provides an optional interface to the vulkan instance, devices and queues.
/// It is not used by any of the other classes directly and so can be safely ignored if Vookoo
/// is embedded in an engine.
/// See https://vulkan-tutorial.com for details of many operations here.
class Framework {
public:

  FrameworkOptions options;

  Framework() = delete; // Disable default constructor

  // Construct a framework containing the instance, a device and one or more queues.
  Framework(const std::string &name, const FrameworkOptions &options_ = FrameworkOptions{}) : options(options_) {
    vku::InstanceMaker im{};
    instance_ = im
      .defaultLayersExtensions()
      .layer("VK_LAYER_KHRONOS_validation")
      //.extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) // added for multiview extension with vulkan 1.0.0
      .apiVersion(VK_MAKE_VERSION(1,1,0))
      .createUnique();

    callback_ = DebugCallback(*instance_);

    auto pds = instance_->enumeratePhysicalDevices();
    physical_device_ = pds[options.deviceID];
    auto qprops = physical_device_.getQueueFamilyProperties();
    const auto badQueue = ~(uint32_t)0;
    graphicsQueueFamilyIndex_ = badQueue;
    computeQueueFamilyIndex_ = badQueue;
    vk::QueueFlags search = vk::QueueFlagBits::eGraphics;
    if (options.useCompute) { search |= vk::QueueFlagBits::eCompute; }

    // Look for an omnipurpose queue family first
    // It is better if we can schedule operations without barriers and semaphores.
    // The Spec says: "If an implementation exposes any queue family that supports graphics operations,
    // at least one queue family of at least one physical device exposed by the implementation
    // must support both graphics and compute operations."
    // Also: All commands that are allowed on a queue that supports transfer operations are
    // also allowed on a queue that supports either graphics or compute operations...
    // As a result we can expect a queue family with at least all three and maybe all four modes.
    std::cout << "Queue Family Properties" << std::endl;
    for (uint32_t qi = 0; qi != qprops.size(); ++qi) {
      auto &qprop = qprops[qi];
      std::cout << "  ID" << qi << " " << vk::to_string(qprop.queueFlags) << "\n";
      if ((qprop.queueFlags & search) == search) {
        graphicsQueueFamilyIndex_ = qi;
        if (options.useCompute) { computeQueueFamilyIndex_ = qi; }
        break;
      }
    }

   if (graphicsQueueFamilyIndex_ == badQueue) {
      std::cout << "oops, missing a graphicsQueue\n";
      return;
    }

    if (options.useCompute && computeQueueFamilyIndex_ == badQueue) {
      std::cout << "oops, missing a computeQueue\n";
      return;
    }

    memprops_ = physical_device_.getMemoryProperties();

    // todo: find optimal texture format
    // auto rgbaprops = physical_device_.getFormatProperties(vk::Format::eR8G8B8A8Unorm);

    vku::DeviceMaker dm{};
    dm.defaultExtensions()
      .queue(graphicsQueueFamilyIndex_)
      .enableGeometryShader( options.useGeometryShader )
      .enableTessellationShader( options.useTessellationShader )
      .enableMultiView( options.useMultiView );
    if (options.useCompute && computeQueueFamilyIndex_ != graphicsQueueFamilyIndex_) dm.queue(computeQueueFamilyIndex_);
    device_ = dm.createUnique(physical_device_);

    vk::PipelineCacheCreateInfo pipelineCacheInfo{};
    pipelineCache_ = device_->createPipelineCacheUnique(pipelineCacheInfo);

    std::vector<vk::DescriptorPoolSize> poolSizes = {
      {vk::DescriptorType::eUniformBuffer, 128},
      {vk::DescriptorType::eCombinedImageSampler, 128},
      {vk::DescriptorType::eStorageBuffer, 128} };

    // Create an arbitrary number of descriptors in a pool.
    // Allow the descriptors to be freed, possibly not optimal behaviour.
    vk::DescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    descriptorPoolInfo.maxSets = 256;
    descriptorPoolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    descriptorPoolInfo.pPoolSizes = poolSizes.data();
    descriptorPool_ = device_->createDescriptorPoolUnique(descriptorPoolInfo);

    ok_ = true;
  }

  void dumpCaps(std::ostream &os) const {
    os << "Memory Types\n";
    for (uint32_t i = 0; i != memprops_.memoryTypeCount; ++i) {
      os << "  type" << i << " heap" << memprops_.memoryTypes[i].heapIndex << " " << vk::to_string(memprops_.memoryTypes[i].propertyFlags) << "\n";
    }
    os << "Heaps\n";
    for (uint32_t i = 0; i != memprops_.memoryHeapCount; ++i) {
      os << "  heap" << vk::to_string(memprops_.memoryHeaps[i].flags) << " " << memprops_.memoryHeaps[i].size << "\n";
    }
  }

  /// Get the Vulkan instance.
  vk::Instance instance() const { return *instance_; }

  /// Get the Vulkan device.
  vk::Device device() const { return *device_; }

  /// Get the queue used to submit graphics jobs
  vk::Queue graphicsQueue() const { return device_->getQueue(graphicsQueueFamilyIndex_, 0); }

  /// Get the queue used to submit compute jobs
  vk::Queue computeQueue() const { return device_->getQueue(computeQueueFamilyIndex_, 0); }

  /// Get the physical device.
  const vk::PhysicalDevice &physicalDevice() const { return physical_device_; }

  /// Get the default pipeline cache (you can use your own if you like).
  vk::PipelineCache pipelineCache() const { return *pipelineCache_; }

  /// Get the default descriptor pool (you can use your own if you like).
  vk::DescriptorPool descriptorPool() const { return *descriptorPool_; }

  /// Get the family index for the graphics queues.
  uint32_t graphicsQueueFamilyIndex() const { return graphicsQueueFamilyIndex_; }

  /// Get the family index for the compute queues.
  uint32_t computeQueueFamilyIndex() const { return computeQueueFamilyIndex_; }

  const vk::PhysicalDeviceMemoryProperties &memprops() const { return memprops_; }

  /// Clean up the framework satisfying the Vulkan verification layers.
  ~Framework() {
    if (device_) {
      device_->waitIdle();
      if (pipelineCache_) {
        pipelineCache_.reset();
      }
      if (descriptorPool_) {
        descriptorPool_.reset();
      }
      device_.reset();
    }

    if (instance_) {
      callback_.reset();
      instance_.reset();
    }
  }

  Framework &operator=(Framework &&rhs) = default;

  /// Returns true if the Framework has been built correctly.
  bool ok() const { return ok_; }

private:
  vk::UniqueInstance instance_;
  vku::DebugCallback callback_;
  vk::UniqueDevice device_;
  //vk::DebugReportCallbackEXT callback_;
  vk::PhysicalDevice physical_device_;
  vk::UniquePipelineCache pipelineCache_;
  vk::UniqueDescriptorPool descriptorPool_;
  uint32_t graphicsQueueFamilyIndex_;
  uint32_t computeQueueFamilyIndex_;
  vk::PhysicalDeviceMemoryProperties memprops_;
  bool ok_ = false;
};

struct WindowOptions
{
  vk::Format desiredSwapChainImageFormat = vk::Format::eB8G8R8A8Unorm;
  vk::PresentModeKHR desiredPresentMode = vk::PresentModeKHR::eFifo;
};

/// This class wraps a window, a surface and a swap chain for that surface.
class Window {
public:

  WindowOptions options;

  Window() = delete; // Disable default constructor

#ifndef VKU_NO_GLFW
  /// Construct a window, surface and swapchain using a GLFW window.
  Window(const vk::Instance &instance, const vk::Device &device, const vk::PhysicalDevice &physicalDevice, uint32_t graphicsQueueFamilyIndex, GLFWwindow *window, const WindowOptions &options_ = WindowOptions{}) : options(options_) {
#ifdef VK_USE_PLATFORM_WIN32_KHR
    auto module = GetModuleHandle(nullptr);
    auto handle = glfwGetWin32Window(window);
    auto ci = vk::Win32SurfaceCreateInfoKHR{{}, module, handle};
    auto surface = instance.createWin32SurfaceKHR(ci);
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
    auto display = glfwGetX11Display();
    auto x11window = glfwGetX11Window(window);
    auto ci = vk::XlibSurfaceCreateInfoKHR{{}, display, x11window};
    auto surface = instance.createXlibSurfaceKHR(ci);
#endif
#ifdef VK_EXT_METAL_SURFACE_EXTENSION_NAME
    vk::SurfaceKHR surface;
    glfwCreateWindowSurface(instance, window,
	                        nullptr,
	                        reinterpret_cast<VkSurfaceKHR *>(&surface));
#endif
    init(instance, device, physicalDevice, graphicsQueueFamilyIndex, surface, options.desiredSwapChainImageFormat);
  }
#endif

  Window(const vk::Instance &instance, const vk::Device &device, const vk::PhysicalDevice &physicalDevice, uint32_t graphicsQueueFamilyIndex, vk::SurfaceKHR surface, const WindowOptions &options_ = WindowOptions{}) : options(options_) {
    init(instance, device, physicalDevice, graphicsQueueFamilyIndex, surface, options.desiredSwapChainImageFormat);
  }

  void init(const vk::Instance &instance, const vk::Device &device, const vk::PhysicalDevice &physicalDevice, uint32_t graphicsQueueFamilyIndex, vk::SurfaceKHR surface, vk::Format desiredSwapChainFormat) {
    surface_ = vk::UniqueSurfaceKHR(surface, vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic>(instance));
    graphicsQueueFamilyIndex_ = graphicsQueueFamilyIndex;
    physicalDevice_ = physicalDevice;
    instance_ = instance;
    device_ = device;
    presentQueueFamily_ = 0;
    auto &pd = physicalDevice;
    auto qprops = pd.getQueueFamilyProperties();
    bool found = false;
    for (uint32_t qi = 0; qi != qprops.size(); ++qi) {
      auto &qprop = qprops[qi];
      if (pd.getSurfaceSupportKHR(qi, surface_.get()) && (qprop.queueFlags & vk::QueueFlagBits::eGraphics) == vk::QueueFlagBits::eGraphics) {
        presentQueueFamily_ = qi;
        found = true;
        break;
      }
    }

    if (!found) {
      std::cout << "No Vulkan present queues found\n";
      return;
    }

    auto fmts = pd.getSurfaceFormatsKHR(surface_.get());
    swapchainImageFormat_ = fmts[0].format;
    swapchainColorSpace_ = fmts[0].colorSpace;
    if (fmts.size() == 1 && swapchainImageFormat_ == vk::Format::eUndefined) {
      swapchainImageFormat_ = vk::Format::eB8G8R8A8Unorm;
      swapchainColorSpace_ = vk::ColorSpaceKHR::eSrgbNonlinear;
    } else {
      for (auto &fmt : fmts) {
        if (fmt.format == desiredSwapChainFormat) {
          swapchainImageFormat_ = fmt.format;
          swapchainColorSpace_ = fmt.colorSpace;
        }
      }
    }

    createSwapchain();

    createImages();

    createDepthStencil();

	  createRenderPass();

	  createFrameBuffers();

    for (int i = 0; i != numImageIndices(); ++i) {
      vk::SemaphoreCreateInfo sci;
      imageAcquireSemaphore_.emplace_back(device.createSemaphoreUnique(sci));
      commandCompleteSemaphore_.emplace_back(device.createSemaphoreUnique(sci));
      dynamicSemaphore_.emplace_back(device.createSemaphoreUnique(sci));
    }

    typedef vk::CommandPoolCreateFlagBits ccbits;

    vk::CommandPoolCreateInfo cpci{ ccbits::eTransient|ccbits::eResetCommandBuffer, graphicsQueueFamilyIndex };
    commandPool_ = device.createCommandPoolUnique(cpci);

    // Create static draw buffers
    vk::CommandBufferAllocateInfo cbai{ *commandPool_, vk::CommandBufferLevel::ePrimary, (uint32_t)framebuffers_.size() };
    staticDrawBuffers_ = device.allocateCommandBuffersUnique(cbai);
    dynamicDrawBuffers_ = device.allocateCommandBuffersUnique(cbai);

    // Create a set of fences to protect the command buffers from re-writing.
    for (int i = 0; i != staticDrawBuffers_.size(); ++i) {
      vk::FenceCreateInfo fci;
      fci.flags = vk::FenceCreateFlagBits::eSignaled;
      commandBufferFences_.emplace_back(device.createFence(fci));
    }

    for (int i = 0; i != staticDrawBuffers_.size(); ++i) {
      vk::CommandBuffer cb = *staticDrawBuffers_[i];
      vk::CommandBufferBeginInfo bi{};
      cb.begin(bi);
      cb.end();
    }

    // Create a set of fences to protect the dynamic command buffers from re-writing.
    for (int i = 0; i != dynamicDrawBuffers_.size(); ++i) {
      vk::FenceCreateInfo fci;
      fci.flags = vk::FenceCreateFlagBits::eSignaled;
      dynamicCommandBufferFences_.emplace_back(device.createFence(fci));
    }

    for (int i = 0; i != dynamicDrawBuffers_.size(); ++i) {
      vk::CommandBuffer cb = *dynamicDrawBuffers_[i];
      vk::CommandBufferBeginInfo bi{};
      cb.begin(bi);
      cb.end();
    }

    ok_ = true;
  }

	/// Dump the capabilities of the physical device used by this window.
  void dumpCaps(std::ostream &os, vk::PhysicalDevice pd) const {
    os << "Surface formats\n";
    auto fmts = pd.getSurfaceFormatsKHR(surface_.get());
    for (auto &fmt : fmts) {
      auto fmtstr = vk::to_string(fmt.format);
      auto cstr = vk::to_string(fmt.colorSpace);
      os << "  format=" << fmtstr << " colorSpace=" << cstr << "\n";
    }

    os << "Present Modes\n";
    auto presentModes = pd.getSurfacePresentModesKHR(surface_.get());
    for (auto pm : presentModes) {
      std::cout << "  " << vk::to_string(pm) << "\n";
    }
  }

  static void defaultRenderFunc(vk::CommandBuffer cb, int currentFrame, vk::RenderPassBeginInfo &rpbi) {
    vk::CommandBufferBeginInfo bi{};
    cb.begin(bi);
    cb.end();
  }

  typedef void (renderFunc_t)(vk::CommandBuffer cb, int currentFrame, vk::RenderPassBeginInfo &rpbi);

  /// Build a static draw buffer. This will be rendered after any dynamic
  /// content generated in draw()
  void setStaticCommands(const std::function<renderFunc_t> &func) {
    this->func = func;
    buildStaticCBs();
  }

  void buildStaticCBs() {
    if(func) {
      for (int i = 0; i != staticDrawBuffers_.size(); ++i) {
        vk::CommandBuffer cb = *staticDrawBuffers_[i];

        vk::ClearDepthStencilValue clearDepthValue{1.0f, 0};
        std::array<vk::ClearValue, 2> clearColours{
            vk::ClearValue{clearColorValue()}, clearDepthValue};
        vk::RenderPassBeginInfo rpbi;
        rpbi.renderPass = *renderPass_;
        rpbi.framebuffer = *framebuffers_[i];
        rpbi.renderArea = vk::Rect2D{{0, 0}, {width_, height_}};
        rpbi.clearValueCount = (uint32_t)clearColours.size();
        rpbi.pClearValues = clearColours.data();

        func(cb, i, rpbi);
      }
    }
  }

  /// Queue the static command buffer for the next image in the swap chain. Optionally call a function to create a dynamic command buffer
  /// for uploading textures, changing uniforms etc.
  void draw(const vk::Device &device, const vk::Queue &graphicsQueue, const std::function<void (vk::CommandBuffer cb, int currentFrame, vk::RenderPassBeginInfo &rpbi)> &dynamic = defaultRenderFunc) {
    /* uncomment to use time ***********************************
    static auto start = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::high_resolution_clock::now();
    auto delta = time - start;
    start = time;
    //std::cout << std::chrono::duration_cast<std::chrono::microseconds>(delta).count() << "us frame time\n";
    *************************************************************/

    auto umax = std::numeric_limits<uint64_t>::max();
    static uint32_t currentFrame = 0;
    uint32_t imageIndex = 0;
    
    vk::Fence &rpcbFence = dynamicCommandBufferFences_[currentFrame];
    {vk::Result result = device.waitForFences(rpcbFence, 1, umax);} // TODO use result
    
    vk::Fence &cbFence = commandBufferFences_[currentFrame];
    {vk::Result result = device.waitForFences(cbFence, 1, umax);} // TODO use result
    
    vk::Semaphore iaSema = *imageAcquireSemaphore_[currentFrame];
    auto acquired = device.acquireNextImageKHR(*swapchain_, umax, iaSema, VK_NULL_HANDLE, &imageIndex);
    if (acquired == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapChain();
      return;
    } else if (acquired != vk::Result::eSuccess && acquired != vk::Result::eSuboptimalKHR) {
      throw std::runtime_error("failed to present swap chain image!");
    }

    device.resetFences(rpcbFence);
    device.resetFences(cbFence);

    vk::CommandBuffer pscb = *dynamicDrawBuffers_[currentFrame];
    vk::Semaphore psSema = *dynamicSemaphore_[currentFrame];

    vk::ClearDepthStencilValue clearDepthValue{ 1.0f, 0 };
    std::array<vk::ClearValue, 2> clearColours{vk::ClearValue{clearColorValue()}, clearDepthValue};
    vk::RenderPassBeginInfo rpbi;
    rpbi.renderPass = *renderPass_;
    rpbi.framebuffer = *framebuffers_[currentFrame];
    rpbi.renderArea = vk::Rect2D{{0, 0}, {width_, height_}};
    rpbi.clearValueCount = (uint32_t)clearColours.size();
    rpbi.pClearValues = clearColours.data();
    dynamic(pscb, currentFrame, rpbi);

    vk::PipelineStageFlags waitStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submit;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &iaSema;
    submit.pWaitDstStageMask = &waitStages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &pscb;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &psSema;

    {vk::Result result = graphicsQueue.submit(1, &submit, rpcbFence);} // TODO use result

    vk::CommandBuffer cb = *staticDrawBuffers_[currentFrame];
    vk::Semaphore ccSema = *commandCompleteSemaphore_[currentFrame];

    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &psSema;
    submit.pWaitDstStageMask = &waitStages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &ccSema;

    {vk::Result result = graphicsQueue.submit(1, &submit, cbFence);} // TODO use result

    vk::PresentInfoKHR presentInfo;
    vk::SwapchainKHR swapchain = *swapchain_;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.swapchainCount = 1;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &ccSema;

    vk::Result resultPresent;
    try { resultPresent = presentQueue().presentKHR(presentInfo); } 
    catch (vk::OutOfDateKHRError err) { resultPresent = vk::Result::eErrorOutOfDateKHR; }
    catch (...) { throw std::runtime_error("failed to present swap chain image!"); }
    if (resultPresent == vk::Result::eErrorOutOfDateKHR || resultPresent == vk::Result::eSuboptimalKHR) {
      recreateSwapChain();
      return;
    }

    ++currentFrame %= numImageIndices();
  }

  /// Return the queue family index used to present the surface to the display.
  uint32_t presentQueueFamily() const { return presentQueueFamily_; }

  /// Get the queue used to submit graphics jobs
  vk::Queue presentQueue() const { return device_.getQueue(presentQueueFamily_, 0); }

  /// Return true if this window was created sucessfully.
  bool ok() const { return ok_; }

  /// Return the renderpass used by this window.
  vk::RenderPass renderPass() const { return *renderPass_; }

  /// Return the frame buffers used by this window
  const std::vector<vk::UniqueFramebuffer> &framebuffers() const { return framebuffers_; }

  /// Destroy resources when shutting down.
  ~Window() {
    for (auto &iv : imageViews_) {
      device_.destroyImageView(iv);
    }
    for (auto &f : commandBufferFences_) {
      device_.destroyFence(f);
    }
    for (auto &f : dynamicCommandBufferFences_) {
      device_.destroyFence(f);
    }
    swapchain_ = vk::UniqueSwapchainKHR{};
  }

  Window &operator=(Window &&rhs) = default;

  /// Return the width of the display.
  uint32_t width() const { return width_; }

  /// Return the height of the display.
  uint32_t height() const { return height_; }

  /// Return the format of the back buffer.
  vk::Format swapchainImageFormat() const { return swapchainImageFormat_; }

  /// Return the colour space of the back buffer (Usually sRGB)
  vk::ColorSpaceKHR swapchainColorSpace() const { return swapchainColorSpace_; }

  /// Return the swapchain object
  vk::SwapchainKHR swapchain() const { return *swapchain_; }

  /// Return the views of the swap chain images
  const std::vector<vk::ImageView> &imageViews() const { return imageViews_; }

  /// Return the swap chain images
  const std::vector<vk::Image> &images() const { return images_; }

  /// Return the static command buffers.
  const std::vector<vk::UniqueCommandBuffer> &commandBuffers() const { return staticDrawBuffers_; }

  /// Return the fences used to control the static buffers.
  const std::vector<vk::Fence> &commandBufferFences() const { return commandBufferFences_; }

  /// Return the fences used to control the dynamic buffers.
  const std::vector<vk::Fence> &dynamicCommandBufferFences() const { return dynamicCommandBufferFences_; }

  /// Return the semaphore signalled when an image is acquired.
  const std::vector<vk::UniqueSemaphore> &imageAcquireSemaphore() const { return imageAcquireSemaphore_; }

  /// Return the semaphore signalled when the command buffers are finished.
  const std::vector<vk::UniqueSemaphore> &commandCompleteSemaphore() const { return commandCompleteSemaphore_; }

  /// Return the semaphore signalled when a presentation is finished.
  const std::vector<vk::UniqueSemaphore> &dynamicSemaphore() const { return dynamicSemaphore_; }

  /// Return a defult command Pool to use to create new command buffers.
  vk::CommandPool commandPool() const { return *commandPool_; }

  /// Return the number of swap chain images.
  int numImageIndices() const { return (int)images_.size(); }

  /// Create a new swapchain and destroy the previous one if any.
  void createSwapchain() {
    auto pms = physicalDevice_.getSurfacePresentModesKHR(surface_.get());
    vk::PresentModeKHR presentMode = pms[0];
    if (std::find(pms.begin(), pms.end(), options.desiredPresentMode) != pms.end()) {
      presentMode = options.desiredPresentMode;
    } else {
      std::cout << "No present mode available\n";
      return;
    }

    auto surfaceCaps = physicalDevice_.getSurfaceCapabilitiesKHR(surface_.get());
    if (width_ != surfaceCaps.currentExtent.width || height_ != surfaceCaps.currentExtent.height) {
      width_ = surfaceCaps.currentExtent.width;
      height_ = surfaceCaps.currentExtent.height;
    }
    vk::SwapchainCreateInfoKHR swapinfo{};
    std::array<uint32_t, 2> queueFamilyIndices = {graphicsQueueFamilyIndex_,
                                                  presentQueueFamily_};
    bool sameQueues = queueFamilyIndices[0] == queueFamilyIndices[1];
    vk::SharingMode sharingMode = !sameQueues ? vk::SharingMode::eConcurrent
                                              : vk::SharingMode::eExclusive;
    swapinfo.imageExtent = surfaceCaps.currentExtent;
    swapinfo.surface = surface_.get();
    swapinfo.minImageCount = surfaceCaps.minImageCount + 1;
    swapinfo.imageFormat = swapchainImageFormat_;
    swapinfo.imageColorSpace = swapchainColorSpace_;
    swapinfo.imageExtent = surfaceCaps.currentExtent;
    swapinfo.imageArrayLayers = 1;
    swapinfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    swapinfo.imageSharingMode = sharingMode;
    swapinfo.queueFamilyIndexCount = !sameQueues ? 2 : 0;
    swapinfo.pQueueFamilyIndices = queueFamilyIndices.data();
    swapinfo.preTransform = surfaceCaps.currentTransform;
 
    swapinfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapinfo.presentMode = presentMode;
    swapinfo.clipped = 1;
    swapinfo.oldSwapchain = *swapchain_;
    swapchain_ = device_.createSwapchainKHRUnique(swapinfo);
  }

  void createImages() {
    images_ = device_.getSwapchainImagesKHR(*swapchain_);
    for (auto &iv : imageViews_) {
      device_.destroyImageView(iv);
    }
    imageViews_.clear();
    for (auto &img : images_) {
      vk::ImageViewCreateInfo ci{};
      ci.image = img;
      ci.viewType = vk::ImageViewType::e2D;
      ci.format = swapchainImageFormat_;
      ci.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
      imageViews_.emplace_back(device_.createImageView(ci));
    }
  }

  void createFrameBuffers() {
    framebuffers_.clear();
    for (int i = 0; i != imageViews_.size(); ++i) {
      vk::ImageView attachments[2] = {imageViews_[i],
                                      depthStencilImage_.imageView()};
      vk::FramebufferCreateInfo fbci{{},     *renderPass_, 2, attachments,
                                     width_, height_,      1};
      framebuffers_.push_back(device_.createFramebufferUnique(fbci));
    }
  }

  void createDepthStencil() {
    auto memprops = physicalDevice_.getMemoryProperties();
    depthStencilImage_ =
        vku::DepthStencilImage(device_, memprops, width_, height_);
  }

  void createRenderPass() { // Build the renderpass using two attachments,
                            // colour and depth/stencil.
    RenderpassMaker rpm;

    // The only colour attachment.
    rpm.attachmentBegin(swapchainImageFormat_);
    rpm.attachmentLoadOp(vk::AttachmentLoadOp::eClear);
    rpm.attachmentStoreOp(vk::AttachmentStoreOp::eStore);
    rpm.attachmentFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    // The depth/stencil attachment.
    rpm.attachmentBegin(depthStencilImage_.format());
    rpm.attachmentLoadOp(vk::AttachmentLoadOp::eClear);
    rpm.attachmentStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
    rpm.attachmentFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    // A subpass to render using the above two attachments.
    rpm.subpassBegin(vk::PipelineBindPoint::eGraphics);
    rpm.subpassColorAttachment(vk::ImageLayout::eColorAttachmentOptimal, 0);
    rpm.subpassDepthStencilAttachment(
        vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);

    // A dependency to reset the layout of both attachments.
    rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
    rpm.dependencySrcStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    rpm.dependencyDstStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    rpm.dependencyDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                                vk::AccessFlagBits::eColorAttachmentWrite);

    // A dependency to reset the layout of DepthStencil.
    rpm.dependencyBegin(VK_SUBPASS_EXTERNAL, 0);
    rpm.dependencySrcStageMask(
        vk::PipelineStageFlagBits::eLateFragmentTests);
    rpm.dependencyDstStageMask(
        vk::PipelineStageFlagBits::eEarlyFragmentTests);
    rpm.dependencySrcAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentWrite);
    rpm.dependencyDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                vk::AccessFlagBits::eDepthStencilAttachmentWrite);

    // Use the maker object to construct the vulkan object
    renderPass_ = rpm.createUnique(device_);
  }

  void recreateSwapChain() {
    device_.waitIdle();

    createSwapchain();
    createImages();
    createDepthStencil();
    createFrameBuffers();
    buildStaticCBs();
  }

  vk::Device device() const { return device_; }

  std::array<float,4> &clearColorValue() { return clearColorValue_; }

private:
  vk::Instance instance_;
  vk::PhysicalDevice physicalDevice_;
  uint32_t graphicsQueueFamilyIndex_;
  vk::UniqueSurfaceKHR surface_;
  vk::UniqueSwapchainKHR swapchain_;
  vk::UniqueRenderPass renderPass_;
  std::vector<vk::UniqueSemaphore> imageAcquireSemaphore_;
  std::vector<vk::UniqueSemaphore> commandCompleteSemaphore_;
  std::vector<vk::UniqueSemaphore> dynamicSemaphore_;
  vk::UniqueCommandPool commandPool_;

  std::vector<vk::ImageView> imageViews_;
  std::vector<vk::Image> images_;
  std::vector<vk::Fence> commandBufferFences_;
  std::vector<vk::Fence> dynamicCommandBufferFences_;
  std::vector<vk::UniqueFramebuffer> framebuffers_;
  std::vector<vk::UniqueCommandBuffer> staticDrawBuffers_;
  std::vector<vk::UniqueCommandBuffer> dynamicDrawBuffers_;
  /// \brief Function called to recreate the static buffers on window size
  /// change.
  std::function<renderFunc_t> func;

  vku::DepthStencilImage depthStencilImage_;

  uint32_t presentQueueFamily_ = 0;
  uint32_t width_;
  uint32_t height_;
  std::array<float, 4> clearColorValue_{0.75f, 0.75f, 0.75f, 1};
  vk::Format swapchainImageFormat_ = vk::Format::eB8G8R8A8Unorm;
  vk::ColorSpaceKHR swapchainColorSpace_ = vk::ColorSpaceKHR::eSrgbNonlinear;
  vk::Device device_;
  bool ok_ = false;
};

} // namespace vku

#endif // VKU_FRAMEWORK_HPP
