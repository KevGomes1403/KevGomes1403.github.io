+++
date = '2026-05-06T00:00:00-04:00'
draft = false
title = 'Implementing a low-latency megakernel for Qwen3 MoE on Trainium 3'
categories = ["technical"]
+++

<style>
.post-content table:not(.highlighttable, .highlight table, .gist .highlight) {
  width: fit-content;
  max-width: 100%;
  margin-left: auto;
  margin-right: auto;
}
</style>

AWS Trainium ASICs are some of the highest volume inference accelerations deployed today alongside NVIDIA GPUs and Google TPUs. Anthropic has notably committed a significant amount of FLOPs to Trainium 2 and 3, collaborating with AWS on Project Rainier. Despite competitive perf/TCO and peak theoretical performance, a nascent software ecosystem leaves it relatively uncompetitive in the landscape of inference accelerators.

Trainium is primarily programmed through a Python DSL called Neuron Kernel Interface (NKI) which makes low level ISA instructions for arithmetic ops, data movement, and collectives available to the programmer. Unlike CUDA, where the programming model involves coordinating many small Streaming Multiprocessors, optimizing Trainium kernels requires keeping a smaller number of large compute engines utilized and overlapping data movement with compute through careful DMA scheduling. See [this post](https://www.linkedin.com/posts/emi-andere_anthropic-runs-claude-on-over-a-million-trainium2-share-7443033866522521600-hRcr/) for a much more nuanced understanding of Trainium programming.

The other half of the Trainium software stack is XLA, which handles graph compilation for the majority of a model's compute. In practice, a deployed model is a mix of XLA subgraphs and NKI kernels. The problem is that XLA and NKI subgraphs don’t share memory. NKI kernel inputs must reside in HBM, so every transition between an XLA subgraph and an NKI kernel forces a round-trip to HBM, even if the data was just computed and could have stayed on-chip.

## Why Megakernels?

As explored in [recent work on GPU megakernels](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), writing individual kernels is problematic for high-throughput inference since kernel launch and teardown overheads leave the accelerator underutilized between ops, hurting end-to-end throughput. The work highlights three key barriers to writing highly fused kernels on CUDA:

>
>
> 1. Fusing dozens of operations is hard to do from scratch. We need a
> mechanism for executing these operations within the megakernel.
> 2. In order to overlap multiple operations on the same hardware, we
> need to prevent contention over limited resources, such as shared
> memory.
> 3. The GPU synchronizes after each kernel in the traditional kernel
> model. Without kernels, we have to synchronize the GPU all by ourselves

Trainium is the right target for maximal fusion since the Neuron compiler handles low level scheduling and memory management. Stanford’s megakernel for Llama on NVIDIA H100 and B100 GPUs uses a Python interpreter for scheduling pre-fused operations ahead-of-time and dispatching them to Streaming Multiprocessors. A well-written NKI program encodes the operation order directly; the Neuron compiler then handles instruction scheduling, engine assignments, and synchronization insertion, removing the need for a runtime dispatcher.

On the memory management front, NKI constrains the first dimension of each Sbuf tensor to 128, otherwise the programmer is free to logically allocate as many tensors as they’d like. However, should overflows happen, the compiler automatically inserts spill/reload instructions that temporarily evict data to HBM and load them back when needed. This improves programmability but requires careful engineering to prevent performance hits.

Data dependencies are also tracked by the compiler, removing the need for explicit synchronization by the programmer.

Programmability and generalizability remain key challenges to megakernel design. Every LLM is different, some like DeepSeek employing novel attention mechanisms like sparse attention, others like Gemma 3 including per-layer embeddings. Each new model release necessitates manual performance engineering and revamping kernel libraries to optimize for specific use cases. Trainium already has a configurable kernel library, supporting a wide variety of popular kernel variations. With recent support for collectives in Sbuf in the `nki.collectives` module, megakernels have become a viable option for running inference on Trainium.

NKI is expressive enough that a single engineer can write a full-model megakernel, and the compiler handles enough of the manual engineering work done on GPUs that this is tractable

## Megakernel Design

The megakernel performs all 48 layers of Qwen 3 MoE for decode (token gen), including two AllReduce operations, with the exceptions of the input embedding and LM Head layers. The design goals were modularity, removing unnecessary HBM roundtrips, and persisting most of the model in a single NKI jit context to expose cross-kernel optimizations to the compiler. This kernel targets bs=1, seq=640 — extending to larger batches and contexts increases attention/MoE complexity but uses the same NKI primitives.

On-chip memory allocation is done using **SbufManager,** a set of APIs provided by AWS for stack and heap management of on-chip data. It also provides convenient multi-buffering support using the `interleave_degree` parameter which overlaps the weight fetch for the next layer while the current layer is computing. Fixed Sbuf addresses are assigned to hidden states as the compiler requires them for performing AllReduce. The output of each layer is persisted in Sbuf for the next layer to immediately use instead of being written back to HBM.

I wrote and optimized each of the subkernels for attention and MoE in isolation. Logical Neuron Cores (LNC) config was enabled such that two physical neuron cores are combined into a logical one. This required efficient sharding strategies within a single kernel with sendrecv instructions to gather intermediates within a LNC. Moreover, it allows for Qwen’s 4 KV heads to be sharded across 4 LNC’s in a single Trainium 3 chip.

1. Attention subkernel fuses RMSNorm, Q/K/V projections, KV cache updates, self-attention, and output projection.
2. MoE subkernel also fuses the input RMSNorm in addition to the router and experts gemm operations. I use selective weight loading which is efficient for single batch execution.

## Agent-Assisted Micro-Optimization

I built three Claude Code skills to assist with the performance engineering loop, each targeting a distinct phase:

**Debugging** (`nki-kernel-debug`): Validates NKI kernels against a compiled Neuron reference. The workflow starts with `nki.simulate` on CPU: no hardware needed, fast iteration, simulating with bf16-faithful numerics. When hardware and simulator disagree, the skill escalates to HLO inspection to identify the exact dtype transition or op-ordering issue.

**Optimization** (`nki-kernel-optimizer`): Runs an explicit orchestrator/subagent loop. The orchestrator classifies the bottleneck from profiling metrics (compute-bound, memory-bound, stall-bound, spill-bound) and generates $N$ independent optimization plans. Subagents are dispatched one at a time and each runs a correctness loop until `assert_allclose` passes before benchmarking. Results are reported as structured metrics (device time, dma active time, spilled bytes etc) which the orchestrator synthesizes into the next round.

**Profiling** (`neuron-profile`): Goes beyond aggregate engine utilization to per-instruction source attribution. The profiler cli emits a per-instruction table tagged with source NKI references, which lets the agent pinpoint which NKI source lines contribute to DMA stalls or idle gaps. This moves the agent from making vague claims about bottlenecks to citing exact profiling stats.

Each skill embeds curated reference pages from the AWS Neuron docs — DMA patterns, SBUF allocation, LNC sharding, trn3 architecture — rather than relying on Claude's base training. This made the difference between agents that hallucinate Neuron APIs and agents that converge quickly.

The kernel generation and optimization workflow was: profile → agent classifies bottleneck and generates plans → subagents implement and validate autonomously → I review benchmark deltas and pick the best variant → re-profile. The correctness loop being agent-owned was the key leverage point. The agent handles the tedious cycle of writing correct code, compiling, testing, and fixing that otherwise dominates kernel work.

## Results

I left all compiler flags equivalent across implementations with on-device sampling enabled so that all performance differences are attributed to the kernels used. The megakernel targets decode (token-generation) since it is memory-bound and benefits from any latency savings. I’m comparing it against the library model available in Trainium’s NeuronX Distributed Inference library which implements the entire decode phase in PyTorch XLA. Then, I enabled the MoE kernel available in NKI Library while keeping attention in XLA (the attention kernel didn’t support this sequence length).

{{< figure src="throughput.svg" alt="throughput" align="center" width="80%" >}}

The megakernel saves ~5ms of decode stage latency compared to the baseline, giving it a 1.76x throughput boost. AWS’s own MoE kernel optimizes the baseline quite a bit but leaves room for more improvement. For the remainder of this post, I will compare the megakernel model to the baseline NKI+XLA hybrid.

Digging deeper into the profile, the per-layer latency improves by ~30 us (18.5%). While the average per-layer compute engine times are comparable, the synchronization and DMA times drop significantly.  The −34.5 μs/layer sync-engine saving is per-kernel-launch sync engine setup that runs on each physical NeuronCore on every graph boundary. In the baseline, this cost is paid for about 98 separate graph breaks in the decoder block.

At the same time, the AllReduce overhead doubles due to the attention subkernel computing the full output on both physical cores per LNC. This doubles the effective tensor capacity being reduced. This can be mitigated in a future version of the attention subkernel through better LNC sharding.

| Metric | NKI+XLA | Megakernel | Δ |
| --- | --- | --- | --- |
| Wall time | 165.26 μs | 134.47 μs | **−30.80 μs** |
| DMA active | 91.99 μs | 66.50 μs | −25.49 μs |
| Sync engine | 53.52 μs | 19.05 μs | **−34.47 μs** |
| Scalar engine | 41.21 μs | 24.06 μs | −17.15 μs |
| GPSIMD engine | 38.65 μs | 26.84 μs | −11.80 μs |
| Tensor engine | 48.86 μs | 48.15 μs | −0.71 μs |
| Vector engine | 49.37 μs | 57.95 μs | +8.58 μs |
| CC ops (ARs) | 13.45 μs | 27.77 μs | +14.32 μs |

The improvement in DMA time came from the reduction in the number of DMA packets. Trainium’s DMA engines are responsible for moving data between HBM and Sbuf, or within HBM and Sbuf themselves. DMA throughput improves as the amount of data transferred increases. Smaller transfers suffer from packet generation overheads while larger ones amortize the costs and better saturate memory bandwidth.

{{< figure src="nki-dma-intro-1.jpg" alt="dma" align="center" width="80%" caption="Source: [AWS Neuron NKI DMA overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/nki-dma-overview.html)" >}}

The profile shows that the megakernel has **43% fewer DMA packets** with each packet being **2.7× larger** on average. Moreover, for weight loading, the baseline emits ~24 small packets per weight load (~374 B each); megakernel emits ~8.5 (~1,012 B each). For the same logical operation, there are ~3× fewer descriptors to compute.

|  | NKI+XLA | Megakernel |
| --- | --- | --- |
| LDWEIGHTS instructions | 68,494 | 63,648 |
| HW-dynamic packets | 1,665,656 | 538,464 |
| **Packets per LDWEIGHTS** | **24.3** | **8.5** |
| Total DMA packets | **2,032,870** | **1,156,849** |

## Conclusion

The megakernel approach on Trainium 3 works well because NKI leaves much of the low-level scheduling, synchronization, and memory management work to the compiler. At the same time, it exposes ISA instructions and offers sufficient expressiveness to allow meaningful low-level performance engineering.

Agents are a meaningful productivity multiplier for performance work on Trainium. Performance on trainium accelerators is held back by its relatively nascent software stack. Kernel generation and profiling are well suited to automation by agents, accelerating development on the Neuron platform and helping Trainium close the performance gap to competing inference ASICs. Megakernels will become a viable means to running inference on Trainium as agents lower the barrier to writing high-performance kernels for emerging model architectures.
