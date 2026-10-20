// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include <cstddef>
#include <memory>

#include "mori/core/transport/rdma/proxy/proxy_types.hpp"

namespace mori::core {

// HIP-compiled boundary around KIWI's host invoke API. This prevents its
// device-intrinsic headers from leaking into MORI's ordinary C++ units.
class KiwiProxyHostContext {
 public:
  KiwiProxyHostContext(ProxyRing* ring, size_t capacityLines);
  ~KiwiProxyHostContext();

  KiwiProxyHostContext(const KiwiProxyHostContext&) = delete;
  KiwiProxyHostContext& operator=(const KiwiProxyHostContext&) = delete;

  void* DeviceContext() const;
  void Progress(size_t maxMessages);
  uint8_t CallbackId() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mori::core
