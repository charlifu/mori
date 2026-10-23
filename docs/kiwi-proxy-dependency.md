# KIWI proxy dependency

The optional `BUILD_KIWI_PROXY` backend consumes only KIWI's queue and invoke
core. CMake fetches it with `FetchContent`, pinned to
`8b6b9f4be38c27b00b89b70fa91b561a051af917` from
`https://github.com/AMD-RAD/kiwi.git`. P2P, EP, LCI, and KIWI application
topology support are not enabled.

The queue and invoke files descend from AMD-RAD/hcqueue and retain its MIT
license. A copy is stored in `LICENSES/kiwi-hcqueue.txt`. MORI links KIWI as a
separate build dependency; no KIWI application source is copied into MORI.

Configure MORI with `-DBUILD_KIWI_PROXY=ON`, then select the command transport
at runtime:

```bash
MORI_ENABLE_HOST_PROXY=1 MORI_HOST_PROXY_IMPL=kiwi <application>
```

Omitting `MORI_HOST_PROXY_IMPL`, or setting it to `legacy`, retains the
original MORI ring command path.
The KIWI backend applies to wave- and block-cooperative SHMEM proxy operations;
scalar/divergent `Shmem*Thread` operations deliberately continue to use the
legacy ring as a compatibility fallback. Both paths use MORI's existing
ibverbs QPs, registrations, CQ handling, atomics, multi-NIC routing, and
completion slots. Startup logs report which backend was selected.
