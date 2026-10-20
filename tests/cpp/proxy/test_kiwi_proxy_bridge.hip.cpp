// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License

#include <cassert>
#include <cstring>

#include <hip/hip_runtime.h>

#include "mori/core/transport/rdma/proxy/kiwi_proxy_device.hpp"
#include "mori/core/transport/rdma/proxy/kiwi_proxy_host.hpp"

using namespace mori::core;

__global__ void PostTestCommand(ProxyRing* ring, KiwiProxyDeviceState state,
                                uint64_t marker) {
  KiwiProxyCmd cmd{};
  cmd.op = PROXY_RDMA_WRITE;
  cmd.qp_idx = 3;
  cmd.src_addr = marker;
  cmd.length = 64;
  KiwiProxyPostCommandWave(ring, state, cmd);
}

static void PostAndCheck(ProxyRing* ring, KiwiProxyHostContext& context,
                         uint32_t sequence, uint64_t marker) {
  ring->gpu_head = sequence;
  const uint32_t slot = sequence & PROXY_RING_MASK;
  ring->cmds[slot].status = PROXY_FREE;

  KiwiProxyDeviceState state{};
  state.context = context.DeviceContext();
  state.callbackId = context.CallbackId();
  hipLaunchKernelGGL(PostTestCommand, dim3(1), dim3(64), 0, 0, ring, state,
                     marker);
  assert(hipDeviceSynchronize() == hipSuccess);

  context.Progress(64);
  assert(ring->gpu_head == sequence + 1);
  assert(ring->cmds[slot].status == PROXY_PENDING);
  assert(ring->cmds[slot].op == PROXY_RDMA_WRITE);
  assert(ring->cmds[slot].qp_idx == 3);
  assert(ring->cmds[slot].src_addr == marker);
  assert(ring->cmds[slot].length == 64);
}

int main() {
  ProxyRing* ring = nullptr;
  assert(hipHostMalloc(reinterpret_cast<void**>(&ring), sizeof(ProxyRing),
                       hipHostMallocMapped) == hipSuccess);
  std::memset(ring, 0, sizeof(*ring));

  {
    KiwiProxyHostContext context(ring, 1024);
    PostAndCheck(ring, context, 0, 0x1234);
    PostAndCheck(ring, context, PROXY_RING_SIZE + 5, 0x5678);
  }

  assert(hipHostFree(ring) == hipSuccess);
  return 0;
}
