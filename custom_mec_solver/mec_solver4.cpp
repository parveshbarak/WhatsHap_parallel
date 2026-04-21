/*
 * mec_solver4.cpp  –  plain-MEC diploid solver, version 4
 *
 * Adds sqrt-checkpoint memory management over mec_solver3.
 *
 * mec_solver3 memory: O(N × 2^R)  – stores the full dp table and backtrace
 *                                    table for all N columns.
 * mec_solver4 memory: O(√N × 2^R) – stores only one dp column every
 *                                    k = ceil(√N) steps (checkpoints).
 *                                    No backtrace table at all.
 *
 * How backtrace works without a stored backtrace table:
 *   Divide the N columns into segments of length k.  For each segment,
 *   recompute the dp and backtrace pointers on the fly from the checkpoint
 *   at the segment's left end, then step back through them.  After tracing
 *   back through the segment, discard its temporary arrays and move on to
 *   the next segment to the left.
 *
 * Total time:  O(N × 2^R)  – forward pass once, each column recomputed at
 *              most once more during backtrace (constant factor ≤ 2×).
 * Peak memory: O(√N × 2^R) – checkpoints (permanent) + one segment worth of
 *              temporary dp/bt arrays (freed after each segment).
 *
 * Build: g++ -O2 -std=c++17 -o mec_solver4 mec_solver4.cpp
 * Run:   ./mec_solver4 [path/to/mec_matrix.txt]
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ── Constants ──────────────────────────────────────────────────────────────────

static const int      NO_ENTRY = -1;
static const unsigned INF      = numeric_limits<unsigned>::max() / 2;

// ── Data structures ────────────────────────────────────────────────────────────

struct ReadEntry { int col, allele, quality; };

struct Read {
    string            name;
    vector<ReadEntry> entries;
};

struct MecMatrix {
    int              num_reads;
    int              num_positions;
    vector<int>      positions;
    vector<Read>     reads;
    vector<vector<int>> allele_mat;   // [read][pos]  0/1 or NO_ENTRY
    vector<vector<int>> quality_mat;  // [read][pos]  quality or 0
};

struct ColInfo { vector<int> read_ids; };

// ── Parsing ────────────────────────────────────────────────────────────────────

MecMatrix parse_matrix(const string& filename)
{
    ifstream in(filename);
    if (!in) { cerr << "Cannot open " << filename << "\n"; exit(1); }

    MecMatrix M;
    in >> M.num_reads >> M.num_positions;
    M.positions.resize(M.num_positions);
    for (int i = 0; i < M.num_positions; ++i) in >> M.positions[i];
    in.ignore();

    M.reads.resize(M.num_reads);
    M.allele_mat .assign(M.num_reads, vector<int>(M.num_positions, NO_ENTRY));
    M.quality_mat.assign(M.num_reads, vector<int>(M.num_positions, 0));

    for (int r = 0; r < M.num_reads; ++r) {
        string line;
        getline(in, line);
        istringstream ss(line);
        ss >> M.reads[r].name;
        int col, allele, qual;
        while (ss >> col >> allele >> qual) {
            M.reads[r].entries.push_back({col, allele, qual});
            M.allele_mat [r][col] = allele;
            M.quality_mat[r][col] = qual;
        }
    }
    return M;
}

// ── Column info ────────────────────────────────────────────────────────────────

vector<ColInfo> build_column_info(const MecMatrix& M)
{
    vector<ColInfo> cols(M.num_positions);
    for (int r = 0; r < M.num_reads; ++r)
        for (auto& e : M.reads[r].entries)
            cols[e.col].read_ids.push_back(r);
    for (auto& ci : cols)
        sort(ci.read_ids.begin(), ci.read_ids.end());
    return cols;
}

// ── col_cost (used only in haplotype reconstruction, not in the hot DP loop) ───

unsigned col_cost(int c, uint32_t b, const ColInfo& ci, const MecMatrix& M)
{
    unsigned q00=0, q01=0, q10=0, q11=0;
    for (int i = 0; i < (int)ci.read_ids.size(); ++i) {
        int r = ci.read_ids[i];
        int allele = M.allele_mat[r][c], qual = M.quality_mat[r][c];
        if (b & (1u<<i)) { if (allele==0) q10+=qual; else q11+=qual; }
        else             { if (allele==0) q00+=qual; else q01+=qual; }
    }
    return min(q00,q01) + min(q10,q11);
}

// ── Gray-code column cost precomputation ──────────────────────────────────────
//
// Returns costs[b] for all b in [0, 2^Rc).
// O(Rc) seed + O(1) per subsequent bipartition = O(Rc + 2^Rc) total.

vector<unsigned> precompute_costs_gray(int c, const ColInfo& ci, const MecMatrix& M)
{
    int Rc = ci.read_ids.size();
    uint32_t B = 1u << Rc;
    vector<unsigned> costs(B);

    unsigned q00=0, q01=0, q10=0, q11=0;
    for (int i = 0; i < Rc; ++i) {
        int r = ci.read_ids[i];
        int allele = M.allele_mat[r][c], qual = M.quality_mat[r][c];
        if (allele == 0) q00 += qual; else q01 += qual;
    }
    costs[0] = min(q00,q01);  // partition 1 empty

    uint32_t cur_b = 0;
    for (uint32_t n = 1; n < B; ++n) {
        int f = __builtin_ctz(n);
        int r = ci.read_ids[f];
        int allele = M.allele_mat[r][c], qual = M.quality_mat[r][c];
        if (cur_b & (1u << f)) {
            if (allele == 0) { q10 -= qual; q00 += qual; }
            else             { q11 -= qual; q01 += qual; }
        } else {
            if (allele == 0) { q00 -= qual; q10 += qual; }
            else             { q01 -= qual; q11 += qual; }
        }
        cur_b ^= (1u << f);
        costs[cur_b] = min(q00,q01) + min(q10,q11);
    }
    return costs;
}

// ── Single DP column transition ────────────────────────────────────────────────
//
// Computes dp_curr (column c) from dp_prev (column c-1).
// If bt is non-null, fills bt[bc] = argmin b_prev that transitions to bc.
// bt is used only during backtrace segment recomputation; passing nullptr
// during the forward pass avoids allocating the backtrace vector.

void dp_step(const vector<unsigned>& dp_prev,
             int c,
             const vector<ColInfo>& cols,
             const MecMatrix& M,
             vector<unsigned>& dp_curr,
             vector<uint32_t>* bt)
{
    auto& ids_prev = cols[c-1].read_ids;
    auto& ids_curr = cols[c  ].read_ids;
    int Rp = ids_prev.size(), Rc = ids_curr.size();

    // ── Shared reads ────────────────────────────────────────────────────────────
    vector<pair<int,int>> shared;
    for (int i=0, j=0; i<Rp && j<Rc; ) {
        if      (ids_prev[i] == ids_curr[j]) { shared.push_back({i,j}); ++i; ++j; }
        else if (ids_prev[i] <  ids_curr[j]) ++i;
        else                                 ++j;
    }

    uint32_t shared_p = 0, shared_c = 0;
    for (auto& [i,j] : shared) { shared_p |= (1u<<i); shared_c |= (1u<<j); }

    // ── Projection table (size 2^Rp, key = bp & shared_p) ─────────────────────
    vector<unsigned>  proj       (1u<<Rp, INF);
    vector<uint32_t>  proj_argmin(1u<<Rp, 0);
    for (uint32_t bp = 0; bp < (1u<<Rp); ++bp) {
        uint32_t key = bp & shared_p;
        if (dp_prev[bp] < proj[key]) {
            proj       [key] = dp_prev[bp];
            proj_argmin[key] = bp;
        }
    }

    // ── Remap table (size 2^Rc, O(2^Rc) via recurrence) ──────────────────────
    // remap[bc] translates shared bits of bc from curr to prev bit positions.
    vector<uint32_t> single_remap(Rc, 0);
    for (auto& [i,j] : shared) single_remap[j] = (1u<<i);

    vector<uint32_t> remap(1u<<Rc, 0);
    for (uint32_t x = 1; x < (1u<<Rc); ++x) {
        int lsb = __builtin_ctz(x);
        remap[x] = remap[x^(1u<<lsb)] | single_remap[lsb];
    }

    // ── Column costs via Gray code ─────────────────────────────────────────────
    auto costs_c = precompute_costs_gray(c, cols[c], M);

    // ── DP update ──────────────────────────────────────────────────────────────
    dp_curr.assign(1u<<Rc, INF);
    if (bt) bt->assign(1u<<Rc, 0);

    for (uint32_t bc = 0; bc < (1u<<Rc); ++bc) {
        uint32_t key  = remap[bc];
        unsigned prev = proj[key];
        if (prev < INF) {
            dp_curr[bc] = prev + costs_c[bc];
            if (bt) (*bt)[bc] = proj_argmin[key];
        }
    }
}

// ── Sqrt-checkpoint forward pass ───────────────────────────────────────────────
//
// Runs the full forward DP, saving the dp vector at columns 0, k, 2k, …
// Returns the final dp vector (column N-1) separately (not in checkpoints
// unless N-1 happens to be a multiple of k).

vector<unsigned> forward_pass(const MecMatrix& M,
                               const vector<ColInfo>& cols,
                               int k,
                               vector<vector<unsigned>>& ckpts)
{
    int N = M.num_positions;
    int num_ckpts = (N - 1) / k + 1;   // columns 0, k, 2k, … all ≤ N-1
    ckpts.resize(num_ckpts);

    vector<unsigned> dp = precompute_costs_gray(0, cols[0], M);
    ckpts[0] = dp;

    for (int c = 1; c < N; ++c) {
        vector<unsigned> next;
        dp_step(dp, c, cols, M, next, nullptr);
        if (c % k == 0) ckpts[c/k] = next;   // c/k < num_ckpts by construction
        dp = move(next);
    }
    return dp;   // dp at column N-1
}

// ── Sqrt-checkpoint backtrace ──────────────────────────────────────────────────
//
// Recovers the full path (one bipartition per column) given the checkpoints
// and the final dp vector.
//
// For each segment [seg_start, c] (seg_start is a checkpoint column):
//   1. Recompute dp columns seg_start+1 … c from the checkpoint.  While
//      doing so, also compute backtrace pointers (bt[bc] = argmin prev).
//   2. Trace back from path[c] to path[seg_start] using the bt arrays.
//   3. Discard all temporary arrays for this segment and move left.
//
// Temporary memory per segment: O(k × 2^R) – freed after each segment.

vector<uint32_t> backward_pass(const MecMatrix& M,
                                const vector<ColInfo>& cols,
                                const vector<vector<unsigned>>& ckpts,
                                const vector<unsigned>& final_dp,
                                int k)
{
    int N = M.num_positions;
    vector<uint32_t> path(N);

    path[N-1] = (uint32_t)(min_element(final_dp.begin(), final_dp.end())
                            - final_dp.begin());

    int c = N - 1;
    while (c > 0) {
        int seg_start = (c - 1) / k * k;   // leftmost checkpoint ≤ c-1
        int seg_len   = c - seg_start;      // number of columns to recompute

        // local_dp[i] = dp at column (seg_start + i), i = 0 … seg_len
        // local_bt[i] = bt for the transition into column (seg_start + i),
        //               i = 1 … seg_len  (index 0 is unused)
        vector<vector<unsigned>> local_dp(seg_len + 1);
        vector<vector<uint32_t>> local_bt(seg_len + 1);

        local_dp[0] = ckpts[seg_start / k];

        for (int i = 1; i <= seg_len; ++i)
            dp_step(local_dp[i-1], seg_start + i, cols, M,
                    local_dp[i], &local_bt[i]);

        // Step back through the segment
        for (int i = seg_len; i >= 1; --i)
            path[seg_start + i - 1] = local_bt[i][path[seg_start + i]];

        c = seg_start;
    }
    return path;
}

// ── Haplotype reconstruction ───────────────────────────────────────────────────

pair<vector<int>,vector<int>> get_haplotypes(const MecMatrix& M,
                                              const vector<ColInfo>& cols,
                                              const vector<uint32_t>& path)
{
    int N = M.num_positions;
    vector<int> h0(N,-1), h1(N,-1);
    for (int c = 0; c < N; ++c) {
        auto& ids = cols[c].read_ids;
        if (ids.empty()) continue;
        uint32_t b = path[c];
        unsigned q00=0,q01=0,q10=0,q11=0;
        for (int i = 0; i < (int)ids.size(); ++i) {
            int r=ids[i]; int allele=M.allele_mat[r][c]; int qual=M.quality_mat[r][c];
            if (b&(1u<<i)){ if(allele==0) q10+=qual; else q11+=qual; }
            else           { if(allele==0) q00+=qual; else q01+=qual; }
        }
        h0[c] = (q00<=q01) ? 0 : 1;
        h1[c] = (q10<=q11) ? 0 : 1;
    }
    return {h0, h1};
}

// ── Main ───────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    string filename = (argc > 1) ? argv[1] : "mec_matrix.txt";

    MecMatrix M = parse_matrix(filename);
    cout << "Parsed:  " << M.num_reads << " reads,  "
         << M.num_positions << " positions\n";

    vector<ColInfo> cols = build_column_info(M);
    if (M.num_positions == 0) { cout << "MEC cost: 0\n"; return 0; }

    int N = M.num_positions;
    int k = max(1, (int)ceil(sqrt((double)N)));
    cout << "Checkpoint interval k = " << k
         << "  (checkpoints: " << (N-1)/k+1 << ")\n";

    auto t0 = chrono::high_resolution_clock::now();

    // ── Forward pass ────────────────────────────────────────────────────────────
    vector<vector<unsigned>> ckpts;
    vector<unsigned> final_dp = forward_pass(M, cols, k, ckpts);

    unsigned mec = *min_element(final_dp.begin(), final_dp.end());

    // ── Backtrace ────────────────────────────────────────────────────────────────
    vector<uint32_t> path = backward_pass(M, cols, ckpts, final_dp, k);

    auto t1 = chrono::high_resolution_clock::now();
    cout << "Execution time: "
         << chrono::duration_cast<chrono::milliseconds>(t1-t0).count() << " ms\n";

    cout << "MEC cost: " << mec << "\n";

    auto [h0, h1] = get_haplotypes(M, cols, path);

    int preview = min(60, N);
    cout << "H0 (first " << preview << " pos): ";
    for (int c = 0; c < preview; ++c) cout << (h0[c]<0 ? '.' : (char)('0'+h0[c]));
    cout << "\nH1 (first " << preview << " pos): ";
    for (int c = 0; c < preview; ++c) cout << (h1[c]<0 ? '.' : (char)('0'+h1[c]));
    cout << "\n";

    return 0;
}
