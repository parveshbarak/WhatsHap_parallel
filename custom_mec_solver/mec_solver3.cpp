/*
 * mec_solver3.cpp  –  plain-MEC diploid solver, version 3
 *
 * Over mec_solver2.cpp this adds Gray-code traversal when computing column
 * costs.  In mec_solver2, each call to col_cost() loops over all R active
 * reads → O(R) per bipartition → O(2^R × R) per column.
 *
 * Here we precompute all 2^R costs in Gray-code order:
 *   – Initialize q00/q01/q10/q11 for bipartition 0 in O(R).
 *   – Each subsequent Gray-code step flips exactly one bit, so exactly one
 *     read changes partition.  Update the four counters and recompute the
 *     cost in O(1).
 *   – Total per column: O(R + 2^R) = O(2^R).
 *   – Compare mec_solver2: O(2^R × R) per column.
 *
 * Everything else (projection table, remap table, backtrace) is identical
 * to mec_solver2.
 *
 * Build: g++ -O2 -std=c++17 -o mec_solver3 mec_solver3.cpp
 * Run:   ./mec_solver3 [path/to/mec_matrix.txt]
 */

#include <algorithm>
#include <cassert>
#include <chrono>
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
    vector<ReadEntry> entries;   // sorted by col
};

// Dense [num_reads × num_positions] matrices; NO_ENTRY where a read doesn't cover a position.
struct MecMatrix {
    int              num_reads;
    int              num_positions;
    vector<int>      positions;
    vector<Read>     reads;
    vector<vector<int>> allele_mat;   // [read][pos]  0/1 or NO_ENTRY
    vector<vector<int>> quality_mat;  // [read][pos]  quality or 0
};

// Per-column: sorted list of global read indices active at this column.
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

// ── col_cost (used only for haplotype reconstruction, not in the hot DP loop) ──

unsigned col_cost(int c, uint32_t b, const ColInfo& ci, const MecMatrix& M)
{
    unsigned q00=0, q01=0, q10=0, q11=0;
    for (int i = 0; i < (int)ci.read_ids.size(); ++i) {
        int r     = ci.read_ids[i];
        int allele = M.allele_mat [r][c];
        int qual   = M.quality_mat[r][c];
        if (b & (1u << i)) { if (allele==0) q10+=qual; else q11+=qual; }
        else               { if (allele==0) q00+=qual; else q01+=qual; }
    }
    return min(q00,q01) + min(q10,q11);
}

// ── Gray-code cost precomputation ─────────────────────────────────────────────
//
// Returns costs[b] = MEC cost of bipartition b at column c, for all b in
// [0, 2^Rc).  Total work is O(Rc + 2^Rc) instead of O(Rc × 2^Rc).
//
// Algorithm:
//   1. Seed the accumulators for bipartition 0 (all reads in partition 0).
//   2. For step n = 1 … 2^Rc-1:
//        – The bit that flips is f = ctz(n)  (lowest set bit of n).
//        – One read (at local index f) moves between partitions.
//        – Update two of the four accumulators in O(1).
//        – The current bipartition is cur_b = gray(n) = n ^ (n>>1).
//          We maintain it by XOR-flipping bit f each step.
//        – Store cost into costs[cur_b].

vector<unsigned> precompute_costs_gray(int c, const ColInfo& ci, const MecMatrix& M)
{
    int     Rc = ci.read_ids.size();
    uint32_t B = 1u << Rc;
    vector<unsigned> costs(B);

    // ── Seed: bipartition 0 (all reads in partition 0) ────────────────────────
    unsigned q00=0, q01=0, q10=0, q11=0;
    for (int i = 0; i < Rc; ++i) {
        int r     = ci.read_ids[i];
        int allele = M.allele_mat [r][c];
        int qual   = M.quality_mat[r][c];
        if (allele == 0) q00 += qual; else q01 += qual;
    }
    costs[0] = min(q00,q01);   // partition 1 is empty: min(0,0) = 0

    // ── Incremental updates ───────────────────────────────────────────────────
    uint32_t cur_b = 0;                 // tracks the current Gray-code value
    for (uint32_t n = 1; n < B; ++n) {
        int f = __builtin_ctz(n);       // bit that flips at this Gray-code step
        int r     = ci.read_ids[f];
        int allele = M.allele_mat [r][c];
        int qual   = M.quality_mat[r][c];

        if (cur_b & (1u << f)) {
            // read was in partition 1 → moves to partition 0
            if (allele == 0) { q10 -= qual; q00 += qual; }
            else             { q11 -= qual; q01 += qual; }
        } else {
            // read was in partition 0 → moves to partition 1
            if (allele == 0) { q00 -= qual; q10 += qual; }
            else             { q01 -= qual; q11 += qual; }
        }
        cur_b ^= (1u << f);             // advance to next Gray code
        costs[cur_b] = min(q00,q01) + min(q10,q11);
    }
    return costs;
}

// ── DP forward pass ────────────────────────────────────────────────────────────

void run_dp(const MecMatrix& M, const vector<ColInfo>& cols,
            vector<vector<unsigned>>& dp_table,
            vector<vector<uint32_t>>&  backtrace_table)
{
    int N = M.num_positions;
    dp_table       .resize(N);
    backtrace_table.resize(N);

    // ── Column 0 ────────────────────────────────────────────────────────────────
    int R0 = cols[0].read_ids.size();
    backtrace_table[0].assign(1u << R0, 0);  // no predecessor

    auto costs0 = precompute_costs_gray(0, cols[0], M);
    dp_table[0]  = costs0;                   // dp[0][b] = col_cost(0,b)

    // ── Columns 1 .. N-1 ───────────────────────────────────────────────────────
    for (int c = 1; c < N; ++c) {
        auto& ids_prev = cols[c-1].read_ids;
        auto& ids_curr = cols[c  ].read_ids;
        int Rp = ids_prev.size(), Rc = ids_curr.size();

        // Shared reads.
        vector<pair<int,int>> shared;  // (i_in_prev, j_in_curr)
        for (int i=0, j=0; i<Rp && j<Rc; ) {
            if      (ids_prev[i] == ids_curr[j]) { shared.push_back({i,j}); ++i; ++j; }
            else if (ids_prev[i] <  ids_curr[j]) ++i;
            else                                 ++j;
        }

        // Shared-bit masks.
        uint32_t shared_p = 0, shared_c = 0;
        for (auto& [i, j] : shared) { shared_p |= (1u<<i); shared_c |= (1u<<j); }

        // ── Projection table (size 2^Rp) ────────────────────────────────────────
        vector<unsigned>  proj       (1u << Rp, INF);
        vector<uint32_t>  proj_argmin(1u << Rp, 0);
        for (uint32_t bp = 0; bp < (1u << Rp); ++bp) {
            uint32_t key = bp & shared_p;
            if (dp_table[c-1][bp] < proj[key]) {
                proj       [key] = dp_table[c-1][bp];
                proj_argmin[key] = bp;
            }
        }

        // ── Remap table (size 2^Rc, O(2^Rc) build) ──────────────────────────────
        // remap[bc] = the prev-indexed key that bc maps to for the proj lookup.
        vector<uint32_t> single_remap(Rc, 0);
        for (auto& [i, j] : shared) single_remap[j] = (1u << i);

        vector<uint32_t> remap(1u << Rc, 0);
        for (uint32_t x = 1; x < (1u << Rc); ++x) {
            int lsb = __builtin_ctz(x);
            remap[x] = remap[x ^ (1u << lsb)] | single_remap[lsb];
        }

        // ── Column costs via Gray code (O(Rc + 2^Rc)) ───────────────────────────
        auto costs_c = precompute_costs_gray(c, cols[c], M);

        // ── DP update ───────────────────────────────────────────────────────────
        dp_table[c]       .assign(1u << Rc, INF);
        backtrace_table[c].assign(1u << Rc, 0);
        for (uint32_t bc = 0; bc < (1u << Rc); ++bc) {
            unsigned prev = proj[remap[bc]];
            if (prev < INF) {
                dp_table[c][bc]        = prev + costs_c[bc];  // O(1) lookup
                backtrace_table[c][bc] = proj_argmin[remap[bc]];
            }
        }
    }
}

// ── Backtrace ──────────────────────────────────────────────────────────────────

vector<uint32_t> backtrace(int N,
                            const vector<vector<unsigned>>&  dp_table,
                            const vector<vector<uint32_t>>&  bt_table)
{
    vector<uint32_t> path(N);
    auto& dp_last = dp_table[N-1];
    path[N-1] = (uint32_t)(min_element(dp_last.begin(), dp_last.end()) - dp_last.begin());
    for (int c = N-1; c > 0; --c)
        path[c-1] = bt_table[c][path[c]];
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
        unsigned q00=0, q01=0, q10=0, q11=0;
        for (int i = 0; i < (int)ids.size(); ++i) {
            int r = ids[i]; int allele=M.allele_mat[r][c]; int qual=M.quality_mat[r][c];
            if (b & (1u<<i)) { if (allele==0) q10+=qual; else q11+=qual; }
            else             { if (allele==0) q00+=qual; else q01+=qual; }
        }
        h0[c] = (q00 <= q01) ? 0 : 1;
        h1[c] = (q10 <= q11) ? 0 : 1;
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

    vector<vector<unsigned>> dp_table;
    vector<vector<uint32_t>> bt_table;

    auto t0 = chrono::high_resolution_clock::now();
    run_dp(M, cols, dp_table, bt_table);
    auto t1 = chrono::high_resolution_clock::now();
    cout << "Execution time: "
         << chrono::duration_cast<chrono::milliseconds>(t1-t0).count() << " ms\n";

    unsigned mec = *min_element(dp_table[M.num_positions-1].begin(),
                                dp_table[M.num_positions-1].end());
    cout << "MEC cost: " << mec << "\n";

    vector<uint32_t> path = backtrace(M.num_positions, dp_table, bt_table);
    auto [h0, h1] = get_haplotypes(M, cols, path);

    int preview = min(60, M.num_positions);
    cout << "H0 (first " << preview << " pos): ";
    for (int c = 0; c < preview; ++c) cout << (h0[c]<0 ? '.' : (char)('0'+h0[c]));
    cout << "\nH1 (first " << preview << " pos): ";
    for (int c = 0; c < preview; ++c) cout << (h1[c]<0 ? '.' : (char)('0'+h1[c]));
    cout << "\n";

    return 0;
}
