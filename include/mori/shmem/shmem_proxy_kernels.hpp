// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include "mori/core/transport/rdma/proxy/proxy_device_primitives.hpp"
#include "mori/core/transport/rdma/proxy/kiwi_proxy_device.hpp"
#include "mori/shmem/internal.hpp"

#if defined(__HIPCC__) && defined(MORI_PROXY_ENABLED)

namespace mori {
namespace shmem {

inline __device__ volatile core::ProxyRing* ProxyRingForEp(GpuStates* gs, uint32_t epIndex) {
  int pe = epIndex / gs->numQpPerPe;
  int peerLocal = pe % gs->numNics;
  int nicIdx = (gs->localGpuIdx > peerLocal ? gs->localGpuIdx : peerLocal) % gs->numNics;
  return gs->proxyRings[nicIdx];
}

#ifdef MORI_KIWI_PROXY_ENABLED
inline __device__ const core::KiwiProxyDeviceState& KiwiProxyStateForEp(
    GpuStates* gs, uint32_t epIndex) {
  int pe = epIndex / gs->numQpPerPe;
  int peerLocal = pe % gs->numNics;
  int nicIdx = (gs->localGpuIdx > peerLocal ? gs->localGpuIdx : peerLocal) % gs->numNics;
  return gs->kiwiProxy[nicIdx];
}

inline __device__ void ShmemPutMemNbiKiwiWave(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    int pe, int qpId) {
  if (bytes == 0) return;
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(gs, epIndex);
  const auto& state = KiwiProxyStateForEp(gs, epIndex);
  size_t currentOffset = 0;
  size_t remaining = bytes;
  while (remaining > 0) {
    uint32_t lkey;
    uint32_t rkey;
    uintptr_t srcAddr;
    uintptr_t raddr;
    size_t transferSize;
    if (gs->useVMMHeap) {
      srcAddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset + currentOffset;
      size_t srcChunkSize;
      VmmQueryLocalKey(srcAddr, remaining, lkey, srcChunkSize);
      uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset + currentOffset;
      size_t dstChunkSize;
      VmmQueryRemoteAddr(dstAddr, pe, remaining, raddr, rkey, dstChunkSize);
      transferSize = srcChunkSize < dstChunkSize ? srcChunkSize : dstChunkSize;
    } else {
      lkey = source->lkey;
      srcAddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset + currentOffset;
      raddr = dest->peerPtrs[pe] + destOffset + currentOffset;
      rkey = dest->peerRkeys[pe];
      transferSize = remaining;
    }
    core::KiwiProxyPostWriteWave(ring, state, epIndex, srcAddr, lkey, raddr, rkey,
                                  transferSize);
    remaining -= transferSize;
    currentOffset += transferSize;
  }
}

inline __device__ void ShmemPutMemNbiSignalKiwiWave(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset,
    uint64_t signalValue, int pe, int qpId) {
  if (bytes == 0) return;
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  ShmemPutMemNbiKiwiWave(dest, destOffset, source, sourceOffset, bytes, pe, qpId);
  volatile core::ProxyRing* ring = ProxyRingForEp(gs, epIndex);
  const auto& state = KiwiProxyStateForEp(gs, epIndex);
  uintptr_t sigRaddr = signalDest->peerPtrs[pe] + signalDestOffset;
  uint32_t sigRkey = signalDest->peerRkeys[pe];
  core::IbufHandle& ibuf = gs->rdmaEndpoints[epIndex].atomicIbuf;
  core::KiwiProxyPostSignalWriteWave(ring, state, epIndex, sigRaddr, sigRkey,
                                     signalValue, ibuf.lkey);
}

inline __device__ void ShmemPutSizeImmNbiKiwiWave(
    const application::SymmMemObjPtr dest, size_t destOffset, const void* val,
    size_t bytes, int pe, int qpId) {
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  uintptr_t raddr;
  uint32_t rkey;
  if (gs->useVMMHeap) {
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;
    VmmLookupRemote(dstAddr, pe, raddr, rkey);
  } else {
    raddr = dest->peerPtrs[pe] + destOffset;
    rkey = dest->peerRkeys[pe];
  }
  core::KiwiProxyCmd cmd{};
  cmd.op = core::PROXY_RDMA_WRITE_INLINE;
  cmd.qp_idx = epIndex;
  cmd.dst_addr = raddr;
  cmd.rkey = rkey;
  cmd.inline_tag = core::PROXY_INLINE_SCALAR_WRITE;
  cmd.inline_len = bytes;
  cmd.length = bytes;
  for (size_t off = 0; off < bytes; off += sizeof(uint64_t)) {
    uint64_t word = 0;
    if (__lane_id() == 0) {
      size_t n = bytes - off < sizeof(word) ? bytes - off : sizeof(word);
      memcpy(&word, static_cast<const uint8_t*>(val) + off, n);
    }
    word = __shfl(word, 0);
    memcpy(&cmd.inline_data[off], &word,
           bytes - off < sizeof(word) ? bytes - off : sizeof(word));
  }
  core::KiwiProxyPostCommandWave(ProxyRingForEp(gs, epIndex),
                                  KiwiProxyStateForEp(gs, epIndex), cmd);
}

inline __device__ uint32_t ShmemAtomicNonFetchKiwiWave(
    const application::SymmMemObjPtr dest, size_t destOffset, const void* val,
    size_t bytes, int pe, int qpId) {
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  uintptr_t raddr;
  uint32_t rkey;
  if (gs->useVMMHeap) {
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;
    VmmLookupRemote(dstAddr, pe, raddr, rkey);
  } else {
    raddr = dest->peerPtrs[pe] + destOffset;
    rkey = dest->peerRkeys[pe];
  }
  uint64_t atomicVal = 0;
  if (__lane_id() == 0) memcpy(&atomicVal, val, bytes <= 8 ? bytes : 8);
  atomicVal = __shfl(atomicVal, 0);
  core::IbufHandle& ibuf = gs->rdmaEndpoints[epIndex].atomicIbuf;
  return core::KiwiProxyPostAtomicWave(
      ProxyRingForEp(gs, epIndex), KiwiProxyStateForEp(gs, epIndex), epIndex,
      raddr, rkey, atomicVal, ibuf.lkey, ibuf.addr, false);
}

inline __device__ uint64_t ShmemAtomicFetchKiwiWave(
    const application::SymmMemObjPtr dest, size_t destOffset, const void* val,
    size_t bytes, int pe, int qpId) {
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  uintptr_t raddr;
  uint32_t rkey;
  if (gs->useVMMHeap) {
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;
    VmmLookupRemote(dstAddr, pe, raddr, rkey);
  } else {
    raddr = dest->peerPtrs[pe] + destOffset;
    rkey = dest->peerRkeys[pe];
  }
  uint64_t atomicVal = 0;
  if (__lane_id() == 0) memcpy(&atomicVal, val, bytes <= 8 ? bytes : 8);
  atomicVal = __shfl(atomicVal, 0);
  volatile core::ProxyRing* ring = ProxyRingForEp(gs, epIndex);
  core::IbufHandle& ibuf = gs->rdmaEndpoints[epIndex].atomicIbuf;
  uint32_t seq = core::KiwiProxyPostAtomicWave(
      ring, KiwiProxyStateForEp(gs, epIndex), epIndex, raddr, rkey, atomicVal,
      ibuf.lkey, ibuf.addr, true);
  uint64_t result = 0;
  if (__lane_id() == 0) {
    uint32_t slot = seq & core::PROXY_RING_MASK;
    core::ProxyWaitSlotCompleted(ring, slot);
    result = ring->cmds[slot].result;
  }
  return __shfl(result, 0);
}
#endif

inline __device__ void ShmemQuietAllProxy() {
  GpuStates* gs = GetGlobalGpuStatesPtr();
  for (int n = 0; n < gs->numProxyRings; n++) {
    volatile core::ProxyRing* ring = gs->proxyRings[n];
    if (!ring) continue;
    uint32_t head = ring->gpu_head;
    uint32_t lastQuiet = gs->proxyQuietHead[n];
    if (head != lastQuiet) {
      core::ProxyQuiet(ring, lastQuiet, head - lastQuiet);
      gs->proxyQuietHead[n] = head;
    }
  }
}

// ---------------------------------------------------------------------------
// ShmemQuietThreadKernel<PROXY>
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::PROXY>() {
  ShmemQuietAllProxy();
}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::PROXY>(int pe) {
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe;
  int peerLocal = pe % gs->numNics;
  int nicIdx = (gs->localGpuIdx > peerLocal ? gs->localGpuIdx : peerLocal) % gs->numNics;
  volatile core::ProxyRing* ring = gs->proxyRings[nicIdx];
  if (ring) {
    uint32_t head = ring->gpu_head;
    uint32_t lastQuiet = gs->proxyQuietHead[nicIdx];
    if (head != lastQuiet) {
      core::ProxyQuiet(ring, lastQuiet, head - lastQuiet);
      gs->proxyQuietHead[nicIdx] = head;
    }
  }
}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::PROXY>(int pe, int qpId) {
  ShmemQuietThreadKernel<application::TransportType::PROXY>(pe);
}

// ---------------------------------------------------------------------------
// ShmemPutMemNbiThreadKernel<PROXY> (SymmMemObjPtr)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
  if (bytes == 0) return;
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(gs, epIndex);

  size_t currentOffset = 0;
  size_t remaining = bytes;
  while (remaining > 0) {
    uint32_t lkey;
    uint32_t rkey;
    uintptr_t srcAddr;
    uintptr_t raddr;
    size_t transfer_size;
    if (gs->useVMMHeap) {
      srcAddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset + currentOffset;
      size_t src_chunk_size;
      VmmQueryLocalKey(srcAddr, remaining, lkey, src_chunk_size);
      uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset + currentOffset;
      size_t dst_chunk_size;
      VmmQueryRemoteAddr(dstAddr, pe, remaining, raddr, rkey, dst_chunk_size);
      transfer_size = src_chunk_size < dst_chunk_size ? src_chunk_size : dst_chunk_size;
    } else {
      lkey = source->lkey;
      srcAddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset + currentOffset;
      raddr = dest->peerPtrs[pe] + destOffset + currentOffset;
      rkey = dest->peerRkeys[pe];
      transfer_size = remaining;
    }
    core::ProxyPostWrite(ring, epIndex, srcAddr, lkey, raddr, rkey, transfer_size);
    remaining -= transfer_size;
    currentOffset += transfer_size;
  }
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
#ifdef MORI_KIWI_PROXY_ENABLED
  if (GetGlobalGpuStatesPtr()->kiwiProxyEnabled) {
    ShmemPutMemNbiKiwiWave(dest, destOffset, source, sourceOffset, bytes, pe, qpId);
    return;
  }
#endif
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutMemNbiThreadKernel<application::TransportType::PROXY>(dest, destOffset, source,
                                                                  sourceOffset, bytes, pe, qpId);
  }
}

template <>
inline __device__ void ShmemPutMemNbiBlockKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
  int threadId = core::FlatBlockThreadId();
#ifdef MORI_KIWI_PROXY_ENABLED
  if (GetGlobalGpuStatesPtr()->kiwiProxyEnabled) {
    if (threadId < warpSize) {
      ShmemPutMemNbiKiwiWave(dest, destOffset, source, sourceOffset, bytes, pe, qpId);
    }
    return;
  }
#endif
  if (threadId == 0) {
    ShmemPutMemNbiThreadKernel<application::TransportType::PROXY>(dest, destOffset, source,
                                                                  sourceOffset, bytes, pe, qpId);
  }
}

// ---------------------------------------------------------------------------
// ShmemPutSizeImmNbiThreadKernel<PROXY> (SymmMemObjPtr)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    int qpId) {
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(gs, epIndex);
  uintptr_t raddr;
  uint32_t rkey;
  if (gs->useVMMHeap) {
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;
    VmmLookupRemote(dstAddr, pe, raddr, rkey);
  } else {
    raddr = dest->peerPtrs[pe] + destOffset;
    rkey = dest->peerRkeys[pe];
  }
  core::ProxyPostWriteInline(ring, epIndex, val, 0, raddr, rkey, bytes);
}

template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    int qpId) {
#ifdef MORI_KIWI_PROXY_ENABLED
  if (GetGlobalGpuStatesPtr()->kiwiProxyEnabled) {
    ShmemPutSizeImmNbiKiwiWave(dest, destOffset, val, bytes, pe, qpId);
    return;
  }
#endif
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutSizeImmNbiThreadKernel<application::TransportType::PROXY>(dest, destOffset, val, bytes,
                                                                      pe, qpId);
  }
}

// ---------------------------------------------------------------------------
// ShmemPutMemNbiSignalThreadKernel<PROXY> (SymmMemObjPtr)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(gs, epIndex);
  uint32_t lkey = source->lkey;
  uintptr_t srcAddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset;
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uint32_t rkey = dest->peerRkeys[pe];
  core::ProxyPostWrite(ring, epIndex, srcAddr, lkey, raddr, rkey, bytes);
  uintptr_t sigRaddr = signalDest->peerPtrs[pe] + signalDestOffset;
  uint32_t sigRkey = signalDest->peerRkeys[pe];
  core::IbufHandle& ibuf = gs->rdmaEndpoints[epIndex].atomicIbuf;
  core::ProxyPostSignalWrite(ring, epIndex, sigRaddr, sigRkey, signalValue, ibuf.lkey, ibuf.addr);
}

template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
      dest, destOffset, source, sourceOffset, bytes, signalDest, signalDestOffset, signalValue,
      signalOp, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::PROXY, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
#ifdef MORI_KIWI_PROXY_ENABLED
  if (GetGlobalGpuStatesPtr()->kiwiProxyEnabled) {
    ShmemPutMemNbiSignalKiwiWave(dest, destOffset, source, sourceOffset, bytes, signalDest,
                                 signalDestOffset, signalValue, pe, qpId);
    return;
  }
#endif
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
        dest, destOffset, source, sourceOffset, bytes, signalDest, signalDestOffset, signalValue,
        signalOp, pe, qpId);
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::PROXY, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
#ifdef MORI_KIWI_PROXY_ENABLED
  if (GetGlobalGpuStatesPtr()->kiwiProxyEnabled) {
    ShmemPutMemNbiSignalKiwiWave(dest, destOffset, source, sourceOffset, bytes, signalDest,
                                 signalDestOffset, signalValue, pe, qpId);
    return;
  }
#endif
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
        dest, destOffset, source, sourceOffset, bytes, signalDest, signalDestOffset, signalValue,
        signalOp, pe, qpId);
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::PROXY, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  int threadId = core::FlatBlockThreadId();
#ifdef MORI_KIWI_PROXY_ENABLED
  if (GetGlobalGpuStatesPtr()->kiwiProxyEnabled) {
    if (threadId < warpSize) {
      ShmemPutMemNbiSignalKiwiWave(dest, destOffset, source, sourceOffset, bytes, signalDest,
                                   signalDestOffset, signalValue, pe, qpId);
    }
    return;
  }
#endif
  if (threadId == 0) {
    ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
        dest, destOffset, source, sourceOffset, bytes, signalDest, signalDestOffset, signalValue,
        signalOp, pe, qpId);
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::PROXY, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  int threadId = core::FlatBlockThreadId();
#ifdef MORI_KIWI_PROXY_ENABLED
  if (GetGlobalGpuStatesPtr()->kiwiProxyEnabled) {
    if (threadId < warpSize) {
      ShmemPutMemNbiSignalKiwiWave(dest, destOffset, source, sourceOffset, bytes, signalDest,
                                   signalDestOffset, signalValue, pe, qpId);
    }
    return;
  }
#endif
  if (threadId == 0) {
    ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
        dest, destOffset, source, sourceOffset, bytes, signalDest, signalDestOffset, signalValue,
        signalOp, pe, qpId);
  }
}

// ---------------------------------------------------------------------------
// ShmemAtomicSizeNonFetchThreadKernel<PROXY> (SymmMemObjPtr)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(gs, epIndex);
  uintptr_t raddr;
  uint32_t rkey;
  if (gs->useVMMHeap) {
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;
    VmmLookupRemote(dstAddr, pe, raddr, rkey);
  } else {
    raddr = dest->peerPtrs[pe] + destOffset;
    rkey = dest->peerRkeys[pe];
  }
  core::IbufHandle& ibuf = gs->rdmaEndpoints[epIndex].atomicIbuf;
  uint64_t atomicVal = 0;
  memcpy(&atomicVal, val, bytes <= 8 ? bytes : 8);
  core::ProxyPostAtomicNonFetch(ring, epIndex, raddr, rkey, atomicVal, ibuf.lkey, ibuf.addr);
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
#ifdef MORI_KIWI_PROXY_ENABLED
  if (GetGlobalGpuStatesPtr()->kiwiProxyEnabled) {
    (void)ShmemAtomicNonFetchKiwiWave(dest, destOffset, val, bytes, pe, qpId);
    return;
  }
#endif
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::PROXY>(
        dest, destOffset, val, bytes, amoType, pe, qpId);
  }
}

// ---------------------------------------------------------------------------
// ShmemAtomicTypeFetchThreadKernel<PROXY, T> (SymmMemObjPtr)
// ---------------------------------------------------------------------------
#define DEFINE_PROXY_ATOMIC_FETCH_THREAD(T)                                                      \
  template <>                                                                                    \
  inline __device__ T ShmemAtomicTypeFetchThreadKernel<application::TransportType::PROXY, T>(    \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, void* compare,        \
      size_t bytes, core::atomicType amoType, int pe, int qpId) {                                \
    GpuStates* gs = GetGlobalGpuStatesPtr();                                                     \
    int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);                                 \
    volatile core::ProxyRing* ring = ProxyRingForEp(gs, epIndex);                                \
    uintptr_t raddr;                                                                             \
    uint32_t rkey;                                                                               \
    if (gs->useVMMHeap) {                                                                        \
      uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;              \
      VmmLookupRemote(dstAddr, pe, raddr, rkey);                                                 \
    } else {                                                                                     \
      raddr = dest->peerPtrs[pe] + destOffset;                                                   \
      rkey = dest->peerRkeys[pe];                                                                \
    }                                                                                            \
    core::IbufHandle& ibuf = gs->rdmaEndpoints[epIndex].atomicIbuf;                              \
    uint64_t atomicVal = 0;                                                                      \
    memcpy(&atomicVal, val, bytes <= 8 ? bytes : 8);                                             \
    uint64_t result =                                                                            \
        core::ProxyPostAtomicFetch(ring, epIndex, raddr, rkey, atomicVal, ibuf.lkey, ibuf.addr); \
    T retVal;                                                                                    \
    memcpy(&retVal, &result, sizeof(T));                                                         \
    return retVal;                                                                               \
  }

DEFINE_PROXY_ATOMIC_FETCH_THREAD(uint32_t)
DEFINE_PROXY_ATOMIC_FETCH_THREAD(uint64_t)
DEFINE_PROXY_ATOMIC_FETCH_THREAD(int32_t)
DEFINE_PROXY_ATOMIC_FETCH_THREAD(int64_t)
#undef DEFINE_PROXY_ATOMIC_FETCH_THREAD

#define DEFINE_PROXY_ATOMIC_FETCH_WARP(T)                                                   \
  template <>                                                                               \
  inline __device__ T ShmemAtomicTypeFetchWarpKernel<application::TransportType::PROXY, T>( \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, void* compare,   \
      size_t bytes, core::atomicType amoType, int pe, int qpId) {                           \
    KIWI_PROXY_FETCH_WARP_BODY(T)                                                            \
    return ShmemAtomicTypeFetchThreadKernel<application::TransportType::PROXY, T>(          \
        dest, destOffset, val, compare, bytes, amoType, pe, qpId);                          \
  }

#ifdef MORI_KIWI_PROXY_ENABLED
#define KIWI_PROXY_FETCH_WARP_BODY(T)                                                        \
  if (GetGlobalGpuStatesPtr()->kiwiProxyEnabled) {                                           \
    uint64_t result = ShmemAtomicFetchKiwiWave(dest, destOffset, val, bytes, pe, qpId);      \
    T ret{};                                                                                 \
    memcpy(&ret, &result, sizeof(T));                                                        \
    return ret;                                                                              \
  }
#else
#define KIWI_PROXY_FETCH_WARP_BODY(T)
#endif

DEFINE_PROXY_ATOMIC_FETCH_WARP(uint32_t)
DEFINE_PROXY_ATOMIC_FETCH_WARP(uint64_t)
DEFINE_PROXY_ATOMIC_FETCH_WARP(int32_t)
DEFINE_PROXY_ATOMIC_FETCH_WARP(int64_t)
#undef DEFINE_PROXY_ATOMIC_FETCH_WARP
#undef KIWI_PROXY_FETCH_WARP_BODY

// ---------------------------------------------------------------------------
// ShmemGetMemNbi<PROXY> — not supported (proxy is write-only)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemGetMemNbiThreadKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
  assert(false);
}

template <>
inline __device__ void ShmemGetMemNbiWarpKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
  assert(false);
}

template <>
inline __device__ void ShmemGetMemNbiBlockKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
  assert(false);
}

// ---------------------------------------------------------------------------
// Address-based overloads — stubs (EP uses SymmMemObjPtr APIs, not these)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::PROXY>(
    const void* d, const void* s, size_t b, int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::PROXY>(const void* d,
                                                                                   const void* s,
                                                                                   size_t b, int pe,
                                                                                   int q) {
  assert(false);
}
template <>
inline __device__ void ShmemPutMemNbiBlockKernel<application::TransportType::PROXY>(const void* d,
                                                                                    const void* s,
                                                                                    size_t b,
                                                                                    int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::PROXY>(
    const void* d, void* v, size_t b, int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::PROXY>(
    const void* d, void* v, size_t b, int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::PROXY>(
    const void* d, void* v, size_t b, core::atomicType a, int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::PROXY>(
    const void* d, void* v, size_t b, core::atomicType a, int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemGetMemNbiThreadKernel<application::TransportType::PROXY>(
    void* d, const void* s, size_t b, int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemGetMemNbiWarpKernel<application::TransportType::PROXY>(void* d,
                                                                                   const void* s,
                                                                                   size_t b, int pe,
                                                                                   int q) {
  assert(false);
}
template <>
inline __device__ void ShmemGetMemNbiBlockKernel<application::TransportType::PROXY>(void* d,
                                                                                    const void* s,
                                                                                    size_t b,
                                                                                    int pe, int q) {
  assert(false);
}

// Signal address-based stubs
template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv, core::atomicType so,
    int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, false>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv, core::atomicType so,
    int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::PROXY, true>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv, core::atomicType so,
    int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::PROXY, false>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv, core::atomicType so,
    int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::PROXY, true>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv, core::atomicType so,
    int pe, int q) {
  assert(false);
}
template <>
inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::PROXY, false>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv, core::atomicType so,
    int pe, int q) {
  assert(false);
}

// AtomicFetch address-based stubs
#define DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Scope, T)                                            \
  template <>                                                                                    \
  inline __device__ T ShmemAtomicTypeFetch##Scope##Kernel<application::TransportType::PROXY, T>( \
      const void* d, void* v, void* c, size_t b, core::atomicType a, int pe, int q) {            \
    assert(false);                                                                               \
    return T{};                                                                                  \
  }

DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Thread, uint32_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Thread, uint64_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Thread, int32_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Thread, int64_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Warp, uint32_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Warp, uint64_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Warp, int32_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Warp, int64_t)
#undef DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB

}  // namespace shmem
}  // namespace mori

#endif  // __HIPCC__ && MORI_PROXY_ENABLED
