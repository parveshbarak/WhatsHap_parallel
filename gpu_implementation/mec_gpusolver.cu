/*
 * mec_solver_cuda.cu  —  GPU diploid MEC solver with sqrt-checkpoint backtrace
 *
 * Memory design
 * ─────────────
 * The fundamental problem: exact backtrace requires knowing the argmin
 * decision at every state of every column → O(N × 2^K) storage.
 * For K=20, N=64k: 268 GB.  Streaming to disk gave 170 GB — still unusable.
 *
 * Solution: sqrt-checkpoint divide-and-conquer backtrace (Hirschberg-style).
 *
 *   Forward pass (GPU):
 *     Run the full DP column by column.  Every k = ceil(√N) columns, download
 *     the current DP row and store it as a checkpoint on the host.
 *     Total checkpoint storage: ceil(N/k) × 2^max_R × 4 bytes
 *                             = O(√N × 2^K)
 *     For K=20, N=64k: ~253 checkpoints × 4 MB = ~1 GB.  Fine.
 *     No BT rows stored during the forward pass.
 *
 *   Backward pass (GPU + host):
 *     Divide columns into segments of length k, process right-to-left.
 *     For each segment [seg_start, seg_end]:
 *       1. Re-run the GPU DP from ckpt[seg_start] through seg_end,
 *          this time storing BT rows for each column in the segment.
 *          Peak RAM for one segment: k × 2^max_R × 4 bytes
 *                                  = O(√N × 2^K) ≈ 1 GB.
 *       2. Trace back from path[seg_end] to path[seg_start] using
 *          the stored BT rows.
 *       3. Free the segment's BT rows and move left.
 *
 * Total time:  ~2× the forward pass (each column processed twice).
 * Peak host RAM: O(√N × 2^K) — checkpoints (persistent) + one segment's
 *               BT rows (freed after each segment).
 * For K=20, N=64k: ~2 GB peak host RAM.
 *
 * GPU memory (unchanged):
 *   d_dp_A/B  2 × 2^20 × 4 =  8 MB  (ping-pong DP)
 *   d_bt_col      2^20 × 4 =  4 MB  (single-column BT scratch)
 *   d_proj        2^20 × 8 =  8 MB  (projection table)
 *   d_wq0/1   2 × 2^20 × 4 =  8 MB  (weight LUTs)
 *   Total: 28 MB
 *
 * Build
 * ─────
 *   nvcc -O3 -std=c++17 -arch=sm_80 -o mec_solver_cuda mec_solver_cuda.cu
 *   sm_70=V100  sm_80=A100  sm_86=RTX3090  sm_89=RTX4090
 *
 * Run
 * ───
 *   ./mec_solver_cuda [mec_matrix.txt]
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error checking
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr unsigned INF_COST       = 0x7fffffffu;
static constexpr int      GPU_MAX_R      = 20;   // 2^20 = 1M states; larger → CPU
static constexpr int      BATCH_K_THRESH = 8;    // Rp,Rc ≤ this → small-K kernel
static constexpr int      BLOCK_SIZE     = 256;

// ─────────────────────────────────────────────────────────────────────────────
// Constant memory — shared-read maps (broadcast-cached, kernels C & D)
// ─────────────────────────────────────────────────────────────────────────────

__constant__ uint8_t c_prev_pos[32];
__constant__ uint8_t c_curr_pos[32];

// ─────────────────────────────────────────────────────────────────────────────
// Host data structures
// ─────────────────────────────────────────────────────────────────────────────

struct ReadEntry { int col, allele, quality; };
struct Read      { std::string name; std::vector<ReadEntry> entries; };

struct MecMatrix {
    int num_reads, num_positions;
    std::vector<int>  positions;
    std::vector<Read> reads;
};

struct ColInfo {
    std::vector<int> read_ids;  // sorted, active at this column
    std::vector<int> allele;    // allele[i] for read_ids[i]
    std::vector<int> quality;   // quality[i] for read_ids[i]
};

struct ColDesc {
    int      Rc;
    uint32_t active_mask;
    int      allele [32];
    int      quality[32];
};

struct SmallColDesc {
    int      col_idx;
    int      Rp, Rc, S;
    uint32_t active_mask;
    uint8_t  prev_pos[8];
    uint8_t  curr_pos[8];
    int      allele  [8];
    int      quality [8];
};

// ─────────────────────────────────────────────────────────────────────────────
// Parsing — sparse only, no dense allele_mat/quality_mat
// ─────────────────────────────────────────────────────────────────────────────

static MecMatrix parse_matrix(const std::string& filename)
{
    std::ifstream in(filename);
    if (!in) { std::cerr << "Cannot open " << filename << "\n"; exit(1); }
    MecMatrix M;
    in >> M.num_reads >> M.num_positions;
    M.positions.resize(M.num_positions);
    for (int i = 0; i < M.num_positions; ++i) in >> M.positions[i];
    in.ignore();
    M.reads.resize(M.num_reads);
    for (int r = 0; r < M.num_reads; ++r) {
        std::string line; std::getline(in, line);
        std::istringstream ss(line);
        ss >> M.reads[r].name;
        int col, allele, qual;
        while (ss >> col >> allele >> qual)
            M.reads[r].entries.push_back({col, allele, qual});
    }
    return M;
}

static std::vector<ColInfo> build_column_info(const MecMatrix& M)
{
    std::vector<ColInfo> cols(M.num_positions);
    for (int r = 0; r < M.num_reads; ++r)
        for (auto& e : M.reads[r].entries)
            cols[e.col].read_ids.push_back(r);

    for (int c = 0; c < M.num_positions; ++c) {
        auto& ci = cols[c];
        std::sort(ci.read_ids.begin(), ci.read_ids.end());
        int Rc = (int)ci.read_ids.size();
        ci.allele .resize(Rc, 0);
        ci.quality.resize(Rc, 0);
    }
    for (int r = 0; r < M.num_reads; ++r) {
        for (auto& e : M.reads[r].entries) {
            auto& ci = cols[e.col];
            int pos = (int)(std::lower_bound(ci.read_ids.begin(),
                                             ci.read_ids.end(), r)
                            - ci.read_ids.begin());
            ci.allele [pos] = e.allele;
            ci.quality[pos] = e.quality;
        }
    }
    return cols;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU fallback (Rp or Rc > GPU_MAX_R)
// ─────────────────────────────────────────────────────────────────────────────

static unsigned host_col_cost(uint32_t b, const ColInfo& ci)
{
    unsigned q00=0, q01=0, q10=0, q11=0;
    for (int i = 0; i < (int)ci.read_ids.size(); ++i) {
        int al=ci.allele[i], qu=ci.quality[i];
        if (b & (1u<<i)) { if (al==0) q10+=qu; else q11+=qu; }
        else             { if (al==0) q00+=qu; else q01+=qu; }
    }
    return std::min(q00,q01) + std::min(q10,q11);
}

// Run one DP column transition on the CPU.
// If bt_curr != nullptr, also fills the backtrace row.
static void cpu_dp_column(
    const std::vector<unsigned>& dp_prev,
    const ColInfo& ci_prev, const ColInfo& ci_curr,
    std::vector<unsigned>& dp_curr,
    std::vector<uint32_t>* bt_curr)   // nullptr during forward pass
{
    const auto& ip = ci_prev.read_ids;
    const auto& ic = ci_curr.read_ids;
    int Rp = (int)ip.size(), Rc = (int)ic.size();

    std::vector<std::pair<int,int>> sh;
    for (int i=0, j=0; i<Rp && j<Rc; ) {
        if      (ip[i]==ic[j]) { sh.push_back({i,j}); ++i; ++j; }
        else if (ip[i]< ic[j]) ++i;
        else                   ++j;
    }
    int S = (int)sh.size();
    uint32_t Psz = 1u << S;

    std::vector<unsigned> proj(Psz, INF_COST);
    std::vector<uint32_t> parg(Psz, 0);
    for (uint32_t bp = 0; bp < (1u<<Rp); ++bp) {
        if (dp_prev[bp] == INF_COST) continue;
        uint32_t key = 0;
        for (int k=0; k<S; ++k) if (bp & (1u<<sh[k].first)) key |= (1u<<k);
        if (dp_prev[bp] < proj[key]) { proj[key]=dp_prev[bp]; parg[key]=bp; }
    }
    dp_curr.assign(1u<<Rc, INF_COST);
    if (bt_curr) bt_curr->assign(1u<<Rc, 0);
    for (uint32_t bc = 0; bc < (1u<<Rc); ++bc) {
        uint32_t key = 0;
        for (int k=0; k<S; ++k) if (bc & (1u<<sh[k].second)) key |= (1u<<k);
        if (proj[key] == INF_COST) continue;
        dp_curr[bc] = proj[key] + host_col_cost(bc, ci_curr);
        if (bt_curr) (*bt_curr)[bc] = parg[key];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU Kernels (unchanged from previous version)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void fill_inf(unsigned long long* __restrict__ proj, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) proj[i] = (unsigned long long)INF_COST << 32;
}

__global__ void precompute_wq_layer(
    unsigned* __restrict__ wq0, unsigned* __restrict__ wq1,
    int layer, int allele_l, unsigned qual_l, uint32_t half_sz)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_sz) return;
    uint32_t lo   = tid & ((1u << layer) - 1);
    uint32_t hi   = tid >> layer;
    uint32_t mask = (hi << (layer+1)) | (1u << layer) | lo;
    uint32_t prev = mask ^ (1u << layer);
    wq0[mask] = wq0[prev] + (allele_l == 0 ? qual_l : 0u);
    wq1[mask] = wq1[prev] + (allele_l == 1 ? qual_l : 0u);
}

__global__ void build_proj_kernel(
    const unsigned* __restrict__     dp_prev,
    unsigned long long* __restrict__ proj,
    uint32_t sz_prev, int S)
{
    uint32_t bp = blockIdx.x * blockDim.x + threadIdx.x;
    if (bp >= sz_prev) return;
    unsigned cost = dp_prev[bp];
    if (cost == INF_COST) return;
    uint32_t key = 0;
    for (int k = 0; k < S; ++k)
        if (bp & (1u << c_prev_pos[k])) key |= (1u << k);
    atomicMin(&proj[key],
              ((unsigned long long)cost << 32) | (unsigned long long)bp);
}

__global__ void fill_dp_kernel(
    const unsigned long long* __restrict__ proj,
    const unsigned* __restrict__           wq0,
    const unsigned* __restrict__           wq1,
    unsigned*  __restrict__                dp_curr,
    uint32_t*  __restrict__                bt_col,   // nullptr → skip BT write
    uint32_t sz_curr, int S, uint32_t active_mask,
    bool write_bt)
{
    uint32_t bc = blockIdx.x * blockDim.x + threadIdx.x;
    if (bc >= sz_curr) return;
    uint32_t key = 0;
    for (int k = 0; k < S; ++k)
        if (bc & (1u << c_curr_pos[k])) key |= (1u << k);
    unsigned long long packed    = proj[key];
    unsigned           prev_cost = (unsigned)(packed >> 32);
    if (prev_cost == INF_COST) {
        dp_curr[bc] = INF_COST;
        if (write_bt) bt_col[bc] = 0;
        return;
    }
    uint32_t hap0 = (~bc) & active_mask;
    uint32_t hap1 =   bc  & active_mask;
    unsigned cost = min(wq0[hap0], wq1[hap0]) + min(wq0[hap1], wq1[hap1]);
    dp_curr[bc] = prev_cost + cost;
    if (write_bt) bt_col[bc] = (uint32_t)(packed & 0xFFFFFFFFull);
}

__global__ void batch_small_k_kernel(
    const SmallColDesc* __restrict__ descs,
    const unsigned* __restrict__     dp_prev_global,
    unsigned*  __restrict__          dp_curr_global,
    uint32_t*  __restrict__          bt_out,    // may be nullptr (skip BT)
    int                              bt_stride,
    bool                             write_bt)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    const SmallColDesc& d = descs[bid];

    __shared__ unsigned           s_dp_prev[256];
    __shared__ unsigned long long s_proj   [256];
    __shared__ unsigned           s_wq0    [256];
    __shared__ unsigned           s_wq1    [256];

    uint32_t sz_prev = 1u << d.Rp;
    uint32_t sz_curr = 1u << d.Rc;
    uint32_t sz_proj = 1u << d.S;

    if ((uint32_t)tid < sz_prev)
        s_dp_prev[tid] = dp_prev_global[tid];
    __syncthreads();

    if ((uint32_t)tid < sz_proj)
        s_proj[tid] = (unsigned long long)INF_COST << 32;
    __syncthreads();

    if ((uint32_t)tid < sz_prev) {
        unsigned cost = s_dp_prev[tid];
        if (cost != INF_COST) {
            uint32_t key = 0;
            for (int k = 0; k < d.S; ++k)
                if (tid & (1u << d.prev_pos[k])) key |= (1u << k);
            atomicMin((unsigned long long*)&s_proj[key],
                      ((unsigned long long)cost << 32) | (unsigned long long)tid);
        }
    }
    __syncthreads();

    if (tid == 0) {
        s_wq0[0] = 0; s_wq1[0] = 0;
        for (uint32_t mask = 1; mask < sz_curr; ++mask) {
            int lo = __ffs((int)mask) - 1;
            uint32_t rest = mask ^ (1u << lo);
            unsigned q = (unsigned)d.quality[lo];
            s_wq0[mask] = s_wq0[rest] + (d.allele[lo] == 0 ? q : 0u);
            s_wq1[mask] = s_wq1[rest] + (d.allele[lo] == 1 ? q : 0u);
        }
    }
    __syncthreads();

    if ((uint32_t)tid < sz_curr) {
        uint32_t bc = (uint32_t)tid;
        uint32_t key = 0;
        for (int k = 0; k < d.S; ++k)
            if (bc & (1u << d.curr_pos[k])) key |= (1u << k);
        unsigned long long packed    = s_proj[key];
        unsigned           prev_cost = (unsigned)(packed >> 32);
        unsigned dp_val = INF_COST;
        uint32_t bt_val = 0;
        if (prev_cost != INF_COST) {
            uint32_t hap0 = (~bc) & d.active_mask;
            uint32_t hap1 =   bc  & d.active_mask;
            dp_val = prev_cost + min(s_wq0[hap0], s_wq1[hap0])
                               + min(s_wq0[hap1], s_wq1[hap1]);
            bt_val = (uint32_t)(packed & 0xFFFFFFFFull);
        }
        dp_curr_global[bc] = dp_val;
        if (write_bt && bt_out)
            bt_out[bid * bt_stride + bc] = bt_val;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

static ColDesc make_col_desc(const ColInfo& ci)
{
    ColDesc cd;
    cd.Rc = (int)ci.read_ids.size();
    cd.active_mask = (1u << cd.Rc) - 1;
    for (int i = 0; i < cd.Rc; ++i) {
        cd.allele [i] = ci.allele [i];
        cd.quality[i] = ci.quality[i];
    }
    return cd;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU device state — persistent across forward and backward passes
// ─────────────────────────────────────────────────────────────────────────────

struct GpuState {
    unsigned*           d_dp_A    = nullptr;
    unsigned*           d_dp_B    = nullptr;
    uint32_t*           d_bt_col  = nullptr;
    unsigned long long* d_proj    = nullptr;
    unsigned*           d_wq0     = nullptr;
    unsigned*           d_wq1     = nullptr;
    SmallColDesc*       d_descs   = nullptr;
    uint32_t*           d_bt_sml  = nullptr;  // small-K BT scratch (1 × 256)

    unsigned* h_dp_row   = nullptr;  // pinned, size MAX_STATES
    uint32_t* h_bt_col   = nullptr;  // pinned, size MAX_STATES

    int MAX_STATES = 0;

    void alloc(int max_states) {
        MAX_STATES = max_states;
        CUDA_CHECK(cudaMalloc(&d_dp_A,   (size_t)max_states * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&d_dp_B,   (size_t)max_states * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&d_bt_col, (size_t)max_states * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_proj,   (size_t)max_states * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_wq0,    (size_t)max_states * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&d_wq1,    (size_t)max_states * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&d_descs,  sizeof(SmallColDesc)));
        CUDA_CHECK(cudaMalloc(&d_bt_sml, 256 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_dp_row, (size_t)max_states * sizeof(unsigned)));
        CUDA_CHECK(cudaMallocHost(&h_bt_col, (size_t)max_states * sizeof(uint32_t)));
    }

    void free_all() {
        cudaFree(d_dp_A);  cudaFree(d_dp_B);   cudaFree(d_bt_col);
        cudaFree(d_proj);  cudaFree(d_wq0);    cudaFree(d_wq1);
        cudaFree(d_descs); cudaFree(d_bt_sml);
        cudaFreeHost(h_dp_row);
        cudaFreeHost(h_bt_col);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Core: run one GPU DP column transition
//
// write_bt = false during forward pass (checkpoints only, no BT needed)
// write_bt = true  during backward pass segment recompute
//
// If write_bt=true, downloads BT row into h_bt_out (caller-provided buffer,
// must have sz_curr entries).  Returns the current dp row size.
//
// d_prev / d_curr are the ping-pong DP device buffers (caller manages swap).
// ─────────────────────────────────────────────────────────────────────────────

static uint32_t gpu_dp_column(
    int c,
    const std::vector<ColInfo>& cols,
    GpuState& G,
    unsigned*  d_prev,
    unsigned*  d_curr,
    bool       write_bt,
    uint32_t*  h_bt_out)   // non-null only when write_bt=true
{
    const auto& ci_prev = cols[c-1];
    const auto& ci_curr = cols[c  ];
    int Rp = (int)ci_prev.read_ids.size();
    int Rc = (int)ci_curr.read_ids.size();

    // Shared-read mapping
    uint8_t h_prev_pos[32], h_curr_pos[32];
    int S = 0;
    for (int i=0, j=0; i<Rp && j<Rc; ) {
        if      (ci_prev.read_ids[i] == ci_curr.read_ids[j]) {
            h_prev_pos[S]=(uint8_t)i; h_curr_pos[S]=(uint8_t)j; ++S; ++i; ++j;
        }
        else if (ci_prev.read_ids[i] < ci_curr.read_ids[j]) ++i;
        else ++j;
    }

    uint32_t sz_curr = 1u << Rc;

    // ── Small-K path ──────────────────────────────────────────────────────────
    if (Rp <= BATCH_K_THRESH && Rc <= BATCH_K_THRESH) {
        SmallColDesc sd;
        sd.col_idx = c; sd.Rp=Rp; sd.Rc=Rc; sd.S=S;
        sd.active_mask = (1u<<Rc)-1;
        std::memcpy(sd.prev_pos, h_prev_pos, S);
        std::memcpy(sd.curr_pos, h_curr_pos, S);
        ColDesc cd = make_col_desc(ci_curr);
        for (int i=0; i<Rc; ++i) { sd.allele[i]=cd.allele[i]; sd.quality[i]=cd.quality[i]; }
        CUDA_CHECK(cudaMemcpy(G.d_descs, &sd, sizeof(SmallColDesc), cudaMemcpyHostToDevice));

        int smem = 256*(4+8+4+4);
        batch_small_k_kernel<<<1, 256, smem>>>(
            G.d_descs, d_prev, d_curr,
            write_bt ? G.d_bt_sml : nullptr,
            256, write_bt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (write_bt) {
            CUDA_CHECK(cudaMemcpy(h_bt_out, G.d_bt_sml,
                                  sz_curr*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        return sz_curr;
    }

    // ── Full GPU path ─────────────────────────────────────────────────────────
    uint32_t sz_prev_k = 1u << Rp;
    uint32_t sz_proj_k = 1u << S;
    ColDesc cd = make_col_desc(ci_curr);

    if (S > 0) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_prev_pos, h_prev_pos, S*sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_curr_pos, h_curr_pos, S*sizeof(uint8_t)));
    }

    fill_inf<<<(sz_proj_k+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(G.d_proj, sz_proj_k);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(G.d_wq0, 0, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(G.d_wq1, 0, sizeof(unsigned)));
    {
        uint32_t half = sz_curr >> 1;
        int grid = (half + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int l = 0; l < Rc; ++l) {
            precompute_wq_layer<<<grid, BLOCK_SIZE>>>(
                G.d_wq0, G.d_wq1, l, cd.allele[l], (unsigned)cd.quality[l], half);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    build_proj_kernel<<<(sz_prev_k+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        d_prev, G.d_proj, sz_prev_k, S);
    CUDA_CHECK(cudaGetLastError());

    fill_dp_kernel<<<(sz_curr+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        G.d_proj, G.d_wq0, G.d_wq1, d_curr,
        write_bt ? G.d_bt_col : nullptr,
        sz_curr, S, cd.active_mask, write_bt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (write_bt) {
        CUDA_CHECK(cudaMemcpy(h_bt_out, G.d_bt_col,
                              sz_curr*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
    return sz_curr;
}

// Upload a DP row from host to d_prev
static void upload_dp(GpuState& G, unsigned* d_dst,
                      const std::vector<unsigned>& row)
{
    CUDA_CHECK(cudaMemcpy(d_dst, row.data(),
                          row.size()*sizeof(unsigned), cudaMemcpyHostToDevice));
}

// Download the current DP row from d_prev into a host vector
static std::vector<unsigned> download_dp(GpuState& G, unsigned* d_src, int sz)
{
    CUDA_CHECK(cudaMemcpy(G.h_dp_row, d_src,
                          sz*sizeof(unsigned), cudaMemcpyDeviceToHost));
    return std::vector<unsigned>(G.h_dp_row, G.h_dp_row + sz);
}

// Bootstrap col 0: dp[b] = col_cost(0, b), no predecessor.
static std::vector<unsigned> bootstrap_col0(
    const std::vector<ColInfo>& cols, GpuState& G, unsigned* d_dst)
{
    const auto& ci = cols[0];
    int Rc = (int)ci.read_ids.size();
    uint32_t sz = 1u << Rc;

    // Precompute wq on host for col 0 (one-off)
    std::vector<unsigned> h_wq0(sz,0), h_wq1(sz,0);
    for (uint32_t mask=1; mask<sz; ++mask) {
        int lo=__builtin_ctz(mask); uint32_t rest=mask^(1u<<lo);
        unsigned q=(unsigned)ci.quality[lo];
        h_wq0[mask]=h_wq0[rest]+(ci.allele[lo]==0?q:0u);
        h_wq1[mask]=h_wq1[rest]+(ci.allele[lo]==1?q:0u);
    }
    CUDA_CHECK(cudaMemcpy(G.d_wq0, h_wq0.data(), sz*sizeof(unsigned), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_wq1, h_wq1.data(), sz*sizeof(unsigned), cudaMemcpyHostToDevice));

    unsigned long long zero = 0ull;  // proj[0] = cost=0, argmin=0; S=0
    CUDA_CHECK(cudaMemcpy(G.d_proj, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice));

    ColDesc cd = make_col_desc(ci);
    int grid = (sz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // write_bt=false: col 0 has no predecessor, BT row unused
    fill_dp_kernel<<<grid, BLOCK_SIZE>>>(
        G.d_proj, G.d_wq0, G.d_wq1, d_dst, nullptr,
        sz, /*S=*/0, cd.active_mask, /*write_bt=*/false);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return download_dp(G, d_dst, (int)sz);
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward pass with sqrt-checkpointing
//
// Runs the full DP, saving the DP row every k columns.
// checkpoints[i] = DP row at column i*k  (i=0..num_ckpts-1)
// Returns the final DP row (column N-1).
// No BT rows stored — O(√N × 2^K) host RAM total.
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<unsigned> forward_pass(
    int N, int k,
    const std::vector<ColInfo>& cols,
    GpuState& G,
    std::vector<std::vector<unsigned>>& checkpoints)  // output
{
    int num_ckpts = (N - 1) / k + 1;
    checkpoints.resize(num_ckpts);

    unsigned* d_prev = G.d_dp_A;
    unsigned* d_curr = G.d_dp_B;

    // Col 0: bootstrap
    std::vector<unsigned> dp0 = bootstrap_col0(cols, G, d_prev);
    checkpoints[0] = dp0;

    for (int c = 1; c < N; ++c) {
        int Rp = (int)cols[c-1].read_ids.size();
        int Rc = (int)cols[c  ].read_ids.size();

        if (Rp > GPU_MAX_R || Rc > GPU_MAX_R) {
            // CPU fallback: download, compute on CPU, upload
            int sz_p = 1 << std::min(Rp, GPU_MAX_R);
            std::vector<unsigned> h_prev = download_dp(G, d_prev, sz_p);
            std::vector<unsigned> h_curr;
            std::vector<uint32_t> dummy_bt;
            cpu_dp_column(h_prev, cols[c-1], cols[c], h_curr, nullptr);
            upload_dp(G, d_curr, h_curr);
        } else {
            gpu_dp_column(c, cols, G, d_prev, d_curr,
                          /*write_bt=*/false, nullptr);
        }
        std::swap(d_prev, d_curr);

        // Save checkpoint every k columns
        if (c % k == 0) {
            int Rc_c = std::min((int)cols[c].read_ids.size(), GPU_MAX_R);
            checkpoints[c/k] = download_dp(G, d_prev, 1 << Rc_c);
        }
    }

    // Final DP row
    int Rc_last = std::min((int)cols[N-1].read_ids.size(), GPU_MAX_R);
    return download_dp(G, d_prev, 1 << Rc_last);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward pass with sqrt-checkpoint segment recompute
//
// For each segment [seg_start, seg_end]:
//   1. Upload ckpt[seg_start] to GPU.
//   2. Re-run GPU DP for columns seg_start+1 .. seg_end, this time storing
//      BT rows in seg_bt (vector of vectors, one per column in the segment).
//      Peak RAM: k × 2^max_R × 4 bytes = O(√N × 2^K).
//   3. Trace back from path[seg_end] to path[seg_start].
//   4. Free seg_bt and move left.
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<uint32_t> backward_pass(
    int N, int k,
    const std::vector<ColInfo>& cols,
    GpuState& G,
    const std::vector<std::vector<unsigned>>& checkpoints,
    const std::vector<unsigned>& final_dp)
{
    std::vector<uint32_t> path(N);
    path[N-1] = (uint32_t)(
        std::min_element(final_dp.begin(), final_dp.end()) - final_dp.begin());

    int c = N - 1;
    while (c > 0) {
        // Segment: columns [seg_start .. c], checkpoint at seg_start
        int seg_start = (c / k) * k;
        // Make sure seg_start < c (it's the last checkpoint strictly left of c)
        if (seg_start == c) seg_start -= k;
        if (seg_start < 0)  seg_start = 0;
        int seg_len = c - seg_start;  // number of columns to recompute

        // ── Upload checkpoint at seg_start ────────────────────────────────────
        const std::vector<unsigned>& ckpt = checkpoints[seg_start / k];
        upload_dp(G, G.d_dp_A, ckpt);

        unsigned* d_prev = G.d_dp_A;
        unsigned* d_curr = G.d_dp_B;

        // ── Re-run DP for this segment, storing BT rows ───────────────────────
        // seg_bt[i] = BT row for column (seg_start + i),  i = 1 .. seg_len
        // seg_bt[0] unused (checkpoint column has no BT to store)
        std::vector<std::vector<uint32_t>> seg_bt(seg_len + 1);

        for (int i = 1; i <= seg_len; ++i) {
            int col = seg_start + i;
            int Rp  = (int)cols[col-1].read_ids.size();
            int Rc  = (int)cols[col  ].read_ids.size();
            uint32_t sz_curr = 1u << std::min(Rc, GPU_MAX_R);
            seg_bt[i].resize(sz_curr);

            if (Rp > GPU_MAX_R || Rc > GPU_MAX_R) {
                int sz_p = 1 << std::min(Rp, GPU_MAX_R);
                std::vector<unsigned> h_prev = download_dp(G, d_prev, sz_p);
                std::vector<unsigned> h_curr;
                cpu_dp_column(h_prev, cols[col-1], cols[col],
                              h_curr, &seg_bt[i]);
                upload_dp(G, d_curr, h_curr);
            } else {
                gpu_dp_column(col, cols, G, d_prev, d_curr,
                              /*write_bt=*/true, seg_bt[i].data());
            }
            std::swap(d_prev, d_curr);
        }

        // ── Trace back through the segment ────────────────────────────────────
        for (int i = seg_len; i >= 1; --i) {
            uint32_t state_curr = path[seg_start + i];
            path[seg_start + i - 1] = seg_bt[i][state_curr];
        }
        // seg_bt freed here (goes out of scope at next iteration)

        c = seg_start;
    }

    return path;
}

// ─────────────────────────────────────────────────────────────────────────────
// Haplotype reconstruction
// ─────────────────────────────────────────────────────────────────────────────

static std::pair<std::vector<int>,std::vector<int>> get_haplotypes(
    int N, const std::vector<ColInfo>& cols,
    const std::vector<uint32_t>& path)
{
    std::vector<int> h0(N,-1), h1(N,-1);
    for (int c = 0; c < N; ++c) {
        const auto& ci = cols[c];
        if (ci.read_ids.empty()) continue;
        uint32_t b = path[c];
        unsigned q00=0, q01=0, q10=0, q11=0;
        for (int i=0; i<(int)ci.read_ids.size(); ++i) {
            int al=ci.allele[i], qu=ci.quality[i];
            if (b&(1u<<i)) { if(al==0) q10+=qu; else q11+=qu; }
            else           { if(al==0) q00+=qu; else q01+=qu; }
        }
        h0[c]=(q00<=q01)?0:1;
        h1[c]=(q10<=q11)?0:1;
    }
    return {h0,h1};
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    std::string filename = (argc > 1) ? argv[1] : "mec_matrix.txt";

    MecMatrix M = parse_matrix(filename);
    std::cout << "Parsed:  " << M.num_reads << " reads,  "
              << M.num_positions << " positions\n";

    std::vector<ColInfo> cols = build_column_info(M);
    if (M.num_positions == 0) { std::cout << "MEC cost: 0\n"; return 0; }

    int N = M.num_positions;
    int k = std::max(1, (int)std::ceil(std::sqrt((double)N)));

    // max_R for GPU buffer sizing
    int max_R = 0;
    for (auto& ci : cols) max_R = std::max(max_R, (int)ci.read_ids.size());
    max_R = std::min(max_R, GPU_MAX_R);
    int MAX_STATES = 1 << max_R;

    {
        int small_K=0, cpu_cols=0;
        for (auto& ci : cols) {
            int r=(int)ci.read_ids.size();
            if (r<=BATCH_K_THRESH) ++small_K;
            if (r>GPU_MAX_R)       ++cpu_cols;
        }
        int num_ckpts = (N-1)/k+1;
        size_t ckpt_mb = (size_t)num_ckpts * MAX_STATES * 4 / (1<<20);
        size_t seg_mb  = (size_t)k * MAX_STATES * 4 / (1<<20);
        std::cout << "N=" << N << "  k=" << k << "  checkpoints=" << num_ckpts
                  << "  max_R=" << max_R << "\n"
                  << "Checkpoint RAM: ~" << ckpt_mb << " MB"
                  << "  Peak segment RAM: ~" << seg_mb << " MB\n"
                  << "small-K cols (≤" << BATCH_K_THRESH << "): " << small_K
                  << "  CPU-fallback (>" << GPU_MAX_R << "): " << cpu_cols << "\n";
    }

    GpuState G;
    G.alloc(MAX_STATES);

    cudaEvent_t ev0, ev1, ev2, ev3;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2)); CUDA_CHECK(cudaEventCreate(&ev3));

    // ── Forward pass ──────────────────────────────────────────────────────────
    std::vector<std::vector<unsigned>> checkpoints;
    CUDA_CHECK(cudaEventRecord(ev0));
    std::vector<unsigned> final_dp = forward_pass(N, k, cols, G, checkpoints);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float fwd_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fwd_ms, ev0, ev1));
    std::cout << "Forward pass: " << fwd_ms << " ms\n";

    unsigned mec = *std::min_element(final_dp.begin(), final_dp.end());
    std::cout << "MEC cost: " << mec << "\n";

    // ── Backward pass ─────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(ev2));
    std::vector<uint32_t> path = backward_pass(N, k, cols, G, checkpoints, final_dp);
    CUDA_CHECK(cudaEventRecord(ev3));
    CUDA_CHECK(cudaEventSynchronize(ev3));

    float bwd_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&bwd_ms, ev2, ev3));
    std::cout << "Backward pass: " << bwd_ms << " ms\n";
    std::cout << "Total DP time: " << (fwd_ms + bwd_ms) << " ms\n";

    G.free_all();

    // ── Haplotypes ────────────────────────────────────────────────────────────
    auto [h0, h1] = get_haplotypes(N, cols, path);
    int preview = std::min(60, N);
    std::cout << "H0 (first " << preview << " pos): ";
    for (int c=0; c<preview; ++c)
        std::cout << (h0[c]<0 ? '.' : (char)('0'+h0[c]));
    std::cout << "\n";
    std::cout << "H1 (first " << preview << " pos): ";
    for (int c=0; c<preview; ++c)
        std::cout << (h1[c]<0 ? '.' : (char)('0'+h1[c]));
    std::cout << "\n";

    CUDA_CHECK(cudaEventDestroy(ev0)); CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaEventDestroy(ev2)); CUDA_CHECK(cudaEventDestroy(ev3));
    return 0;
}
