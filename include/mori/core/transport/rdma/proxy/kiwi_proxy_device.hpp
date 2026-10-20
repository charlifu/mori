// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include "mori/core/transport/rdma/proxy/kiwi_proxy.hpp"
#include "mori/core/transport/rdma/proxy/proxy_device_primitives.hpp"

#if defined(__HIPCC__) && defined(MORI_KIWI_PROXY_ENABLED)
#include <kiwi/invoke/device_context.hip.hpp>
#include <kiwi/queue/queue.hpp>

namespace mori::core {

using KiwiProxyQueue = kiwi::queue::MPSCD2HQueue;
using KiwiProxyInvokeContext = kiwi::invoke::DeviceContext<KiwiProxyQueue>;

inline __device__ uint32_t KiwiProxyReserveSlotWave(volatile ProxyRing* ring) {
  uint32_t seq = 0;
  if (__lane_id() == 0) {
    seq = ProxyReserveSlot(ring);
    const uint32_t slot = seq & PROXY_RING_MASK;
    ProxyWaitSlotFree(ring, slot);
    __hip_atomic_store((uint32_t*)&ring->cmds[slot].status, PROXY_RESERVED,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return __shfl(seq, 0);
}

// All lanes must pass a byte-identical descriptor. The MPSC queue owns command
// visibility; the reserved ProxyCmd slot is used only for completion/result and
// by the host verbs executor after the invoke callback materializes the command.
inline __device__ uint32_t KiwiProxyPostCommandWave(
    volatile ProxyRing* ring, const KiwiProxyDeviceState& state, KiwiProxyCmd cmd) {
  const uint32_t seq = KiwiProxyReserveSlotWave(ring);
  cmd.sequence = seq;
  if (cmd.op == PROXY_ATOMIC_FETCH_ADD &&
      cmd.flags == PROXY_FLAGS_FETCH_REQUIRED) {
    *reinterpret_cast<uint64_t*>(&cmd.inline_data[0]) = seq;
  }
  auto* context = static_cast<KiwiProxyInvokeContext*>(state.context);
  auto handle = context->get_device_handle(0);
  (void)handle.invoke(state.callbackId, cmd);
  return seq;
}

inline __device__ uint32_t KiwiProxyPostWriteWave(
    volatile ProxyRing* ring, const KiwiProxyDeviceState& state, uint32_t qpIdx,
    uint64_t srcAddr, uint32_t lkey, uint64_t dstAddr, uint32_t rkey, uint32_t length) {
  KiwiProxyCmd cmd{};
  cmd.op = PROXY_RDMA_WRITE;
  cmd.qp_idx = qpIdx;
  cmd.src_addr = srcAddr;
  cmd.dst_addr = dstAddr;
  cmd.length = length;
  cmd.lkey = lkey;
  cmd.rkey = rkey;
  cmd.flags = PROXY_FLAGS_DEFAULT;
  return KiwiProxyPostCommandWave(ring, state, cmd);
}

inline __device__ uint32_t KiwiProxyPostSignalWriteWave(
    volatile ProxyRing* ring, const KiwiProxyDeviceState& state, uint32_t qpIdx,
    uint64_t dstAddr, uint32_t rkey, uint64_t value, uint32_t lkey) {
  KiwiProxyCmd cmd{};
  cmd.op = PROXY_SIGNAL_WRITE;
  cmd.qp_idx = qpIdx;
  cmd.dst_addr = dstAddr;
  cmd.length = 8;
  cmd.lkey = lkey;
  cmd.rkey = rkey;
  cmd.atomic_arg = value;
  return KiwiProxyPostCommandWave(ring, state, cmd);
}

inline __device__ uint32_t KiwiProxyPostAtomicWave(
    volatile ProxyRing* ring, const KiwiProxyDeviceState& state, uint32_t qpIdx,
    uint64_t dstAddr, uint32_t rkey, uint64_t addValue, uint32_t lkey,
    uint64_t resultAddr, bool fetchRequired) {
  KiwiProxyCmd cmd{};
  cmd.op = PROXY_ATOMIC_FETCH_ADD;
  cmd.qp_idx = qpIdx;
  cmd.src_addr = resultAddr;
  cmd.dst_addr = dstAddr;
  cmd.length = 8;
  cmd.lkey = lkey;
  cmd.rkey = rkey;
  cmd.atomic_arg = addValue;
  cmd.flags = fetchRequired ? PROXY_FLAGS_FETCH_REQUIRED : PROXY_FLAGS_DEFAULT;
  return KiwiProxyPostCommandWave(ring, state, cmd);
}

}  // namespace mori::core
#endif
