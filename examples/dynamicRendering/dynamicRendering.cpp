#include <vku/vku_framework.hpp>
#include <vku/vku.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

// Demonstrates dynamic rendering (VK_KHR_dynamic_rendering, core in Vulkan 1.3).
//
// Two passes — both use vkCmdBeginRendering/vkCmdEndRendering (no VkRenderPass objects):
//   scene pass : render a rotating RGB triangle into a fixed-size offscreen image
//   post  pass : grayscale the offscreen image to the swapchain (fullscreen triangle)
//
// Image layout transitions are explicit barriers in the command buffer.

static constexpr uint32_t OW = 512;
static constexpr uint32_t OH = 512;

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    const char *title = "dynamicRendering";
    auto glfwwindow = glfwCreateWindow(800, 600, title, nullptr, nullptr);

    {
    vku::FrameworkOptions fo;
    fo.useDynamicRendering  = true;
    fo.useSynchronization2  = true;
    vku::Framework fw{title, fo};
    if (!fw.ok()) { std::cout << "Framework creation failed\n"; exit(1); }

    vk::Device device = fw.device();

    vku::Window window(
        fw.instance(), device, fw.physicalDevice(),
        fw.graphicsQueueFamilyIndex(), glfwwindow,
        { .desiredPresentMode = vk::PresentModeKHR::eFifo }
    );
    if (!window.ok()) { std::cout << "Window creation failed\n"; exit(1); }

    ////////////////////////////////////////////////////////////////////////
    // Offscreen image — fixed OW×OH, format R8G8B8A8Unorm.
    // ColorAttachmentImage has eColorAttachment|eSampled usage.

    static constexpr vk::Format offFmt = vk::Format::eR8G8B8A8Unorm;
    vku::ColorAttachmentImage offscreen{device, fw.memprops(), OW, OH, offFmt};

    // Transition from eUndefined once so validation doesn't warn about the first barrier.
    // (The first frame barrier already handles eUndefined→eColorAttachmentOptimal,
    //  but doing an explicit initial transition is cleaner for tools.)
    vku::executeImmediately(device, window.commandPool(), fw.graphicsQueue(), [&](vk::CommandBuffer cb) {
        const vk::ImageSubresourceRange colorSRR{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        vk::ImageMemoryBarrier2 b{
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            offscreen.image(), colorSRR};
        cb.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(b));
    });

    ////////////////////////////////////////////////////////////////////////
    // Sampler for offscreen → post-process.

    auto linearSampler = vku::SamplerMaker{}
        .magFilter(vk::Filter::eLinear).minFilter(vk::Filter::eLinear)
        .addressModeU(vk::SamplerAddressMode::eClampToEdge)
        .addressModeV(vk::SamplerAddressMode::eClampToEdge)
        .createUnique(device);

    ////////////////////////////////////////////////////////////////////////
    // Descriptor set layout — post pass only (scene has no descriptors).

    auto postDSL = vku::DescriptorSetLayoutMaker{}
        .image(0, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1)
        .createUnique(device);

    auto scenePL = vku::PipelineLayoutMaker{}
        .pushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(float))
        .createUnique(device);

    auto postPL = vku::PipelineLayoutMaker{}
        .descriptorSetLayout(*postDSL)
        .createUnique(device);

    ////////////////////////////////////////////////////////////////////////
    // Descriptor set — post pass samples offscreen.

    auto postSets = vku::DescriptorSetMaker{}.layout(*postDSL).create(device, fw.descriptorPool());

    vku::DescriptorSetUpdater{}
        .beginDescriptorSet(postSets[0])
        .beginImages(0, 0, vk::DescriptorType::eCombinedImageSampler)
        .image(*linearSampler, offscreen.imageView(), vk::ImageLayout::eShaderReadOnlyOptimal)
        .update(device);

    ////////////////////////////////////////////////////////////////////////
    // Shaders.

    vku::ShaderModule sceneVert{device, BINARY_DIR "dynamicRenderingScene.vert.spv"};
    vku::ShaderModule sceneFrag{device, BINARY_DIR "dynamicRenderingScene.frag.spv"};
    vku::ShaderModule postVert {device, BINARY_DIR "dynamicRenderingPost.vert.spv"};
    vku::ShaderModule postFrag {device, BINARY_DIR "dynamicRenderingPost.frag.spv"};

    ////////////////////////////////////////////////////////////////////////
    // Pipelines — no VkRenderPass; colorFormat() builds VkPipelineRenderingCreateInfo internally.

    auto scenePipeline = vku::PipelineMaker{OW, OH}
        .shader(vk::ShaderStageFlagBits::eVertex,   sceneVert)
        .shader(vk::ShaderStageFlagBits::eFragment, sceneFrag)
        .dynamicState(vk::DynamicState::eViewport)
        .dynamicState(vk::DynamicState::eScissor)
        .depthTestEnable(VK_FALSE)
        .colorFormat(offFmt)
        .createUnique(device, fw.pipelineCache(), *scenePL);

    auto postPipeline = vku::PipelineMaker{window.width(), window.height()}
        .shader(vk::ShaderStageFlagBits::eVertex,   postVert)
        .shader(vk::ShaderStageFlagBits::eFragment, postFrag)
        .dynamicState(vk::DynamicState::eViewport)
        .dynamicState(vk::DynamicState::eScissor)
        .depthTestEnable(VK_FALSE)
        .colorFormat(window.swapchainImageFormat())
        .createUnique(device, fw.pipelineCache(), *postPL);

    ////////////////////////////////////////////////////////////////////////
    // Main loop.

    int iFrame = 0;
    while (!glfwWindowShouldClose(glfwwindow) && glfwGetKey(glfwwindow, GLFW_KEY_ESCAPE) != GLFW_PRESS) {
        glfwPollEvents();

        int w, h;
        glfwGetWindowSize(glfwwindow, &w, &h);
        if (w == 0 || h == 0) { continue; }

        float t = (float)iFrame * 0.016f;

        window.draw(device, fw.graphicsQueue(),
            [&](vk::CommandBuffer cb, int imageIndex, vk::RenderPassBeginInfo& /*rpbi*/) {

                cb.begin(vk::CommandBufferBeginInfo{});

                using PS2 = vk::PipelineStageFlagBits2;
                using AC2 = vk::AccessFlagBits2;
                const vk::ImageSubresourceRange colorSRR{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

                // ---- Barrier 1: offscreen eShaderReadOnly → eColorAttachmentOptimal ----
                vk::ImageMemoryBarrier2 bOffWr{
                    PS2::eFragmentShader, AC2::eShaderRead,
                    PS2::eColorAttachmentOutput, AC2::eColorAttachmentWrite,
                    vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                    offscreen.image(), colorSRR};
                cb.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(bOffWr));

                // ---- Scene pass: rotating triangle → offscreen ----
                vku::RenderingMaker{OW, OH}
                    .colorClear(offscreen.imageView(), {0.05f, 0.05f, 0.10f, 1.0f})
                    .beginRendering(cb);

                cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *scenePipeline);
                cb.pushConstants(*scenePL, vk::ShaderStageFlagBits::eVertex, 0, sizeof(float), &t);
                vku::setViewportScissor(cb, OW, OH);
                cb.draw(3, 1, 0, 0);
                cb.endRendering();

                // ---- Barrier 2: offscreen→shaderRead, swapchain eUndefined→colorAttachment ----
                std::array<vk::ImageMemoryBarrier2, 2> b2{{
                    {PS2::eColorAttachmentOutput, AC2::eColorAttachmentWrite,
                     PS2::eFragmentShader, AC2::eShaderRead,
                     vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                     VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                     offscreen.image(), colorSRR},
                    {PS2::eNone, AC2::eNone,
                     PS2::eColorAttachmentOutput, AC2::eColorAttachmentWrite,
                     vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                     VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                     window.images()[imageIndex], colorSRR}
                }};
                cb.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(b2));

                // ---- Post pass: grayscale offscreen → swapchain ----
                vku::RenderingMaker{window.width(), window.height()}
                    .colorAttachment(window.imageViews()[imageIndex])
                    .beginRendering(cb);

                cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *postPipeline);
                cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *postPL, 0, postSets[0], nullptr);
                vku::setViewportScissor(cb, window.width(), window.height());
                cb.draw(3, 1, 0, 0);
                cb.endRendering();

                // ---- Barrier 3: swapchain eColorAttachmentOptimal → ePresentSrcKHR ----
                vk::ImageMemoryBarrier2 bPresent{
                    PS2::eColorAttachmentOutput, AC2::eColorAttachmentWrite,
                    PS2::eNone, AC2::eNone,
                    vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                    window.images()[imageIndex], colorSRR};
                cb.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(bPresent));

                cb.end();
            }
        );

        iFrame++;
    }

    device.waitIdle();
    } // Vulkan objects destroyed before GLFW teardown
    glfwDestroyWindow(glfwwindow);
    glfwTerminate();
    return 0;
}
