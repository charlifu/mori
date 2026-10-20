// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include <cstdint>

#include "mori/core/transport/rdma/proxy/proxy_types.hpp"

namespace mori::core {

// Wire payload carried by KIWI. Completion remains in the legacy ProxyCmd
// status/result words so scalar and wave-cooperative producers share quiet,
// error, and atomic-result semantics during the staged migration.
struct KiwiProxyCmd {
  uint64_t sequence;
  uint32_t op;
  uint32_t qp_idx;
  uint64_t src_addr;
  uint64_t dst_addr;
  uint32_t length;
  uint32_t lkey;
  uint32_t rkey;
  uint32_t flags;
  uint64_t atomic_arg;
  uint64_t atomic_swap;
  uint8_t inline_data[PROXY_MAX_INLINE_DATA];
  uint32_t inline_tag;
  uint32_t inline_len;
};

static_assert(sizeof(KiwiProxyCmd) <= 512, "KIWI proxy descriptor exceeds invoke limit");

struct KiwiProxyDeviceState {
  // Kept opaque so ordinary host C++ translation units do not parse KIWI's
  // device-only queue intrinsics. HIP code casts this to the invoke context.
  void* context{nullptr};
  uint8_t callbackId{0};
  uint8_t reserved[7]{};
};

}  // namespace mori::core
