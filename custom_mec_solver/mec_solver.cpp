/*
 * mec_solver.cpp  –  plain-MEC diploid solver (2 haplotypes, no pedigree)
 *
 * Input:  mec_matrix.txt  (written by WhatsHap's pedigreedptable.cpp)
 *
 * Format:
 *   Line 1 : <num_reads> <num_positions>
 *   Line 2 : space-separated genomic positions
 *   Lines 3+: <read_name>  [<snp_col> <allele> <quality>] ...  (sparse)
 *
 * Build: g++ -O2 -std=c++17 -o mec_solver mec_solver.cpp
 * Run:   ./mec_solver [path/to/mec_matrix.txt]
 */

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

// ── Constants ──────────────────────────────────────────────────────────────────

static const int      NO_ENTRY = -1;
static const unsigned INF      = numeric_limits<unsigned>::max() / 2;

// ── Data structures ────────────────────────────────────────────────────────────

struct ReadEntry { int col, allele, quality; };

struct Read {
    string           name;
    vector<ReadEntry> entries;   // sorted by col
};

// Dense [num_reads × num_positions] matrices; NO_ENTRY where a read doesn't cover a position.
struct MecMatrix {
    int              num_reads;
    int              num_positions;
    vector<int>      positions;          // genomic positions
    vector<Read>     reads;
    vector<vector<int>> allele_mat;      // [read][pos]  0/1 or NO_ENTRY
    vector<vector<int>> quality_mat;     // [read][pos]  quality or 0
};

// Per-column: sorted list of global read indices that are active at this column.
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

// ── Column cost ────────────────────────────────────────────────────────────────
//
// For a bipartition b of the active reads at column c:
//   bit i = 0  →  read cols[c].read_ids[i] is assigned to haplotype 0
//   bit i = 1  →  assigned to haplotype 1
// Each haplotype calls the allele (0 or 1) that minimises total quality mismatch.
// Cost = sum of mismatching qualities for haplotype 0 + sum for haplotype 1.

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

// ── DP forward pass ────────────────────────────────────────────────────────────
//
// dp[c][b]  = minimum total MEC cost through columns 0..c with bipartition b
//             of the active reads at column c.
//
// Transition uses a projection table so that the complexity per column is
// O(2^Rprev + 2^Rcurr) rather than O(2^Rprev × 2^Rcurr).
//
// Projection key  =  the assignment (0/1) of the shared reads, packed into a
// bitmask using the order they appear in the shared[] list.
//
// Memory: O(N × 2^R) where R = max active reads per column.
// For large datasets this can be replaced with the sqrt-checkpoint trick
// (recompute between checkpoints during backtrace) – but the simple version is
// easier to understand and verify.

void run_dp(const MecMatrix& M, const vector<ColInfo>& cols,
            vector<vector<unsigned>>& dp_table,
            vector<vector<uint32_t>>&  backtrace_table)
{
    int N = M.num_positions;
    dp_table       .resize(N);
    backtrace_table.resize(N);

    // ── Column 0: no predecessor ────────────────────────────────────────────────
    int R0 = cols[0].read_ids.size();
    dp_table[0].resize(1u << R0);
    backtrace_table[0].assign(1u << R0, 0);  // sentinel
    for (uint32_t b = 0; b < (1u << R0); ++b)
        dp_table[0][b] = col_cost(0, b, cols[0], M);

    // ── Columns 1 .. N-1 ───────────────────────────────────────────────────────
    for (int c = 1; c < N; ++c) {
        auto& ids_prev = cols[c-1].read_ids;
        auto& ids_curr = cols[c  ].read_ids;
        int Rp = ids_prev.size(), Rc = ids_curr.size();

        // Shared reads: sorted intersection, recording positions in prev and curr.
        vector<pair<int,int>> shared;  // (i_in_prev, j_in_curr)
        for (int i=0, j=0; i<Rp && j<Rc; ) {
            if      (ids_prev[i] == ids_curr[j]) { shared.push_back({i,j}); ++i; ++j; }
            else if (ids_prev[i] <  ids_curr[j]) ++i;
            else                                 ++j;
        }
        int S = shared.size();
        uint32_t Psz = 1u << S;

        // proj[key]        = min dp_prev over all b_prev projecting to key
        // proj_argmin[key] = the b_prev that achieved the minimum
        vector<unsigned>  proj       (Psz, INF);
        vector<uint32_t>  proj_argmin(Psz, 0);

        for (uint32_t bp = 0; bp < (1u << Rp); ++bp) {
            uint32_t key = 0;
            for (int k = 0; k < S; ++k)
                if (bp & (1u << shared[k].first)) key |= (1u << k);
            if (dp_table[c-1][bp] < proj[key]) {
                proj       [key] = dp_table[c-1][bp];
                proj_argmin[key] = bp;
            }
        }

        // Compute dp for column c.
        dp_table[c]       .assign(1u << Rc, INF);
        backtrace_table[c].assign(1u << Rc, 0);

        for (uint32_t bc = 0; bc < (1u << Rc); ++bc) {
            // backward projection of bc onto shared-read indexing
            uint32_t key = 0;
            for (int k = 0; k < S; ++k)
                if (bc & (1u << shared[k].second)) key |= (1u << k);

            unsigned prev = proj[key];
            if (prev < INF) {
                unsigned cost = col_cost(c, bc, cols[c], M);
                dp_table[c][bc]        = prev + cost;
                backtrace_table[c][bc] = proj_argmin[key];
            }
        }
    }
}

// ── Backtrace ──────────────────────────────────────────────────────────────────
//
// Returns path[c] = bipartition bitmask of active reads at column c.

vector<uint32_t> backtrace(int N,
                            const vector<vector<unsigned>>&  dp_table,
                            const vector<vector<uint32_t>>&  bt_table)
{
    vector<uint32_t> path(N);

    // Best bipartition at the last column
    auto& dp_last = dp_table[N-1];
    path[N-1] = (uint32_t)(min_element(dp_last.begin(), dp_last.end()) - dp_last.begin());

    for (int c = N-1; c > 0; --c)
        path[c-1] = bt_table[c][path[c]];

    return path;
}

// ── Haplotype reconstruction ───────────────────────────────────────────────────
//
// Given the bipartition path, determine the called allele for each haplotype
// at each position. Returns (h0, h1) each of length num_positions;
// -1 where no reads cover the position.

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
            int r = ids[i];
            int allele = M.allele_mat [r][c];
            int qual   = M.quality_mat[r][c];
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

    // ── Parse ────────────────────────────────────────────────────────────────────
    MecMatrix M = parse_matrix(filename);
    cout << "Parsed:  " << M.num_reads << " reads,  "
         << M.num_positions << " positions\n";

    // M.allele_mat  and M.quality_mat are now filled:
    //   M.allele_mat [r][c]  = 0 or 1      (NO_ENTRY = -1 if read r doesn't cover pos c)
    //   M.quality_mat[r][c]  = quality      (0 if not covered)

    vector<ColInfo> cols = build_column_info(M);

    if (M.num_positions == 0) { cout << "MEC cost: 0\n"; return 0; }

    // ── DP forward pass ──────────────────────────────────────────────────────────
    vector<vector<unsigned>>  dp_table;
    vector<vector<uint32_t>>  bt_table;

    auto start_time = std::chrono::high_resolution_clock::now();

    run_dp(M, cols, dp_table, bt_table);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Execution time: " << duration.count() << " ms\n";

    // ── MEC cost ─────────────────────────────────────────────────────────────────
    auto& dp_last = dp_table[M.num_positions - 1];
    unsigned mec = *min_element(dp_last.begin(), dp_last.end());
    cout << "MEC cost: " << mec << "\n";

    // ── Backtrace & haplotypes ───────────────────────────────────────────────────
    vector<uint32_t> path = backtrace(M.num_positions, dp_table, bt_table);
    auto [h0, h1] = get_haplotypes(M, cols, path);

    // Print haplotypes (first 60 positions as a sanity check; -1 = no coverage)
    int preview = min(60, M.num_positions);
    cout << "H0 (first " << preview << " pos): ";
    for (int c = 0; c < preview; ++c) cout << (h0[c] < 0 ? '.' : (char)('0'+h0[c]));
    cout << "\n";
    cout << "H1 (first " << preview << " pos): ";
    for (int c = 0; c < preview; ++c) cout << (h1[c] < 0 ? '.' : (char)('0'+h1[c]));
    cout << "\n";

    // ── Full haplotype output ─────────────────────────────────────────────────────
    // Uncomment to write full haplotypes to a file:
    /*
    ofstream out("haplotypes.txt");
    out << "# col  genomic_pos  H0  H1\n";
    for (int c = 0; c < M.num_positions; ++c)
        out << c << " " << M.positions[c] << " " << h0[c] << " " << h1[c] << "\n";
    */

    return 0;
}
