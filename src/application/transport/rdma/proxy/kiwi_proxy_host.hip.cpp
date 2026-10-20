// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License

#include "mori/core/transport/rdma/proxy/kiwi_proxy_host.hpp"

#include <kiwi/invoke/invoke.hpp>

#include "mori/core/transport/rdma/proxy/kiwi_proxy.hpp"

namespace mori::core {

using KiwiHostInvokeContext =
    kiwi::invoke::Context<kiwi::queue::MPSCD2HQueue>;

static void PublishCommand(ProxyRing* ring, KiwiProxyCmd cmd) {
  const uint32_t slot =
      static_cast<uint32_t>(cmd.sequence) & PROXY_RING_MASK;
  volatile ProxyCmd& dst = ring->cmds[slot];
  dst.op = cmd.op;
  dst.qp_idx = cmd.qp_idx;
  dst.src_addr = cmd.src_addr;
  dst.dst_addr = cmd.dst_addr;
  dst.length = cmd.length;
  dst.lkey = cmd.lkey;
  dst.rkey = cmd.rkey;
  dst.flags = cmd.flags;
  dst.atomic_arg = cmd.atomic_arg;
  dst.atomic_swap = cmd.atomic_swap;
  __builtin_memcpy(const_cast<uint8_t*>(dst.inline_data), cmd.inline_data,
                   PROXY_MAX_INLINE_DATA);
  dst.inline_tag = cmd.inline_tag;
  dst.inline_len = cmd.inline_len;
  __atomic_store_n(const_cast<uint32_t*>(&dst.status), PROXY_PENDING,
                   __ATOMIC_RELEASE);
}

struct KiwiProxyHostContext::Impl {
  explicit Impl(ProxyRing* ring, size_t capacityLines)
      : context(1, capacityLines) {
    callback = context.register_callback<KiwiProxyCmd>(
        std::function<void(KiwiProxyCmd)>(
            [ring](KiwiProxyCmd cmd) { PublishCommand(ring, cmd); }));
  }

  KiwiHostInvokeContext context;
  uint8_t callback{0};
};

KiwiProxyHostContext::KiwiProxyHostContext(ProxyRing* ring,
                                           size_t capacityLines)
    : impl_(std::make_unique<Impl>(ring, capacityLines)) {}

KiwiProxyHostContext::~KiwiProxyHostContext() = default;

void* KiwiProxyHostContext::DeviceContext() const {
  return impl_->context.device_context();
}

void KiwiProxyHostContext::Progress(size_t maxMessages) {
  (void)impl_->context.progress(0, maxMessages);
}

uint8_t KiwiProxyHostContext::CallbackId() const { return impl_->callback; }

}  // namespace mori::core
