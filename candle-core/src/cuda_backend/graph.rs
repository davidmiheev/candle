//! CUDA-graph capture helper (item-3 launch-overhead work).
//!
//! Thin wrapper over cudarc's stream-capture API bound to a candle
//! [`CudaDevice`]'s stream. Capture a closure of GPU work once, then
//! replay it with a single `cuGraphLaunch` — eliminating per-kernel launch
//! overhead for shape-static workloads (e.g. a decode step with a
//! preallocated KV cache).
//!
//! Constraints inherited from CUDA graphs: everything inside the capture
//! must run on this device's stream, be shape/address-static across
//! replays, and avoid host synchronization. Stream-ordered allocations
//! made during capture become graph-owned alloc nodes; prefer
//! preallocated buffers.

use crate::cuda_backend::CudaDevice;
use crate::Result;
use cudarc::driver::sys::CUstreamCaptureMode;

pub struct CapturedGraph {
    graph: cudarc::driver::CudaGraph,
}

/// True while a [`CapturedGraph::capture`] closure is running. Lets
/// allocation-sensitive code (e.g. the mmq workspaces) fail loudly instead
/// of corrupting a capture with a mid-capture reallocation.
static CAPTURING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn is_capturing() -> bool {
    CAPTURING.load(std::sync::atomic::Ordering::Relaxed)
}

struct CaptureFlagGuard;
impl Drop for CaptureFlagGuard {
    fn drop(&mut self) {
        CAPTURING.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

impl CapturedGraph {
    /// Record all GPU work issued by `f` on `dev`'s stream into a graph.
    /// `f` should perform one representative iteration (e.g. one decode
    /// step) using only preallocated/stable buffers.
    pub fn capture<F: FnOnce() -> Result<()>>(dev: &CudaDevice, f: F) -> Result<Self> {
        let stream = dev.cuda_stream();
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(crate::Error::wrap)?;
        CAPTURING.store(true, std::sync::atomic::Ordering::Relaxed);
        let _flag = CaptureFlagGuard;
        // Run the workload; on failure, make sure capture mode is exited
        // before propagating so the stream is left usable.
        let run = f();
        let graph = stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        run?;
        let graph = graph
            .map_err(crate::Error::wrap)?
            .ok_or_else(|| crate::Error::Msg("graph capture produced no graph".into()))?;
        Ok(Self { graph })
    }

    /// Replay the captured work (single launch).
    pub fn replay(&self) -> Result<()> {
        self.graph.launch().map_err(crate::Error::wrap)
    }
}
