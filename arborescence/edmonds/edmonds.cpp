#include <vector>
#include <unordered_map>
#include <limits>
#include <random>
#include <chrono>
#include <tuple>
#include <torch/extension.h>

using namespace torch::indexing;

#define idx(batch_idx, i, j, max_n) batch_idx * max_n * max_n + i * max_n + j 

class DisjointSetUnion {
 public:
    explicit DisjointSetUnion(int n_vertices) : parent(n_vertices), rank(n_vertices) {
        for (int i = 0; i != n_vertices; ++i) {
            parent[i] = i;
        }
    }

    DisjointSetUnion(const DisjointSetUnion& rhs) : parent(rhs.parent), rank(rhs.rank) {}

    int find(int elem) const {
        if (elem == parent[elem]) {
            return elem;
        } else {
            return find(parent[elem]);
        }
    }

    void unite(int first, int second) {
        int one = find(first);
        int two = find(second);

        if (one == two) {
            return;
        }

        if (rank[one] < rank[two]) {
            parent[one] = two;
        } else {
            parent[two] = one;
        }

        if (rank[one] == rank[two]) {
            ++rank[one];
        }
    }

    void unite_many(const std::vector<int>& vertices) {
        for (int i = 0; i != vertices.size() - 1; ++i) {
            unite(vertices[i], vertices[i + 1]);
        }
    }

    std::unordered_map<int, std::vector<int>> get_sets() const {
        std::unordered_map<int, std::vector<int>> result;
        for (int i = 0; i != parent.size(); ++i) {
            if (result.find(parent[i]) == result.end()) {
                result.emplace(std::make_pair(parent[i], 
                    std::vector<int>{i}));
            } else {
                result[parent[i]].push_back(i);
            }
        }

        return result;
    }

 private:
    std::vector<int> parent;
    std::vector<int> rank;
};

void dfs_cycle(
    std::unordered_map<int, std::vector<int>>& graph,
    std::unordered_map<int, char>& visited,
    int vertex,
    std::unordered_map<int, int>& parent,
    int& cycle_begin,
    int& cycle_end,
    bool& cycled
    )
{
    visited[vertex] = '1';

    if (cycled) {
        return;
    }

    for (auto iter = graph[vertex].begin(); iter != graph[vertex].end(); ++iter) {
        if (visited[*iter] == '0') {
            if (parent.find(*iter) == parent.end()) {
                parent.emplace(std::make_pair(*iter, vertex));
            } else {
                parent[*iter] = vertex;
            }

            dfs_cycle(graph, visited, *iter, parent, cycle_begin, cycle_end, cycled);
        } else if (visited[*iter] == '1') {
            cycled = true;
            parent[*iter] = vertex;
            cycle_end = vertex;
            cycle_begin = *iter;
        }
    }

    visited[vertex] = '2';
}

std::vector<int> find_cycle(std::unordered_map<int, std::vector<int>>& graph) {
    std::unordered_map<int, char> visited;
    for (auto iter = graph.begin(); iter != graph.end(); ++iter) {
        visited.emplace(std::make_pair(iter->first, '0'));
    }

    std::unordered_map<int, int> parent;

    int cycle_begin = -1;
    int cycle_end = -1;
    std::vector<int> result;
    
    for (auto iter = graph.begin(); iter != graph.end(); ++iter) {
        int cur_vertex = iter->first;
        if (visited[cur_vertex] != '2') {
            bool cycled;
            dfs_cycle(graph, visited, cur_vertex, parent, cycle_begin, cycle_end, cycled);

            if (cycled) {
                int current = cycle_end;
                while (current != cycle_begin) {
                    result.push_back(current);
                    current = parent[current];
                }
                result.push_back(current);
                return result;
            }
        }
    }

    return result;
}

struct MinStats {
    std::vector<int> min_x;
    std::vector<int> min_y;
    std::vector<std::vector<int>> mask;
};

std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>> get_minima(
    const std::vector<float>& weights,
    const DisjointSetUnion& sets,
    int root,
    int max_n,
    int batch_idx,
    int length)
{
    std::unordered_map<int, float> min_value;
    std::unordered_map<int, int> min_x;
    std::unordered_map<int, int> min_y;

    int set_root = sets.find(root);

    for (int i = 0; i != length; ++i) {
        for (int j = 0; j != length; ++j) {
            int set_i = sets.find(i);
            int set_j = sets.find(j);

            if (set_j == set_root) {
                continue;
            }

            float value = weights[idx(batch_idx, i, j, max_n)];
            if (set_i != set_j) {
                if (min_value.find(set_j) == min_value.end()) {
                    min_value.emplace(std::make_pair(set_j, value));
                    min_x.emplace(std::make_pair(set_j, i));
                    min_y.emplace(std::make_pair(set_j, j));
                } else if (value < min_value[set_j]) {
                    min_value[set_j] = value;
                    min_x[set_j] = i;
                    min_y[set_j] = j;
                }
            }
        }
    }

    return std::make_pair(min_x, min_y);
}

std::unordered_map<int, std::vector<int>> get_min_graph(
    const std::unordered_map<int, int>& min_x,
    const DisjointSetUnion& sets)
{
    std::unordered_map<int, std::vector<int>> result;

    for (auto iter = min_x.begin(); iter != min_x.end(); ++iter) {
        int v_set_to = iter->first;
        int v_from = iter->second;
        int v_set_from = sets.find(v_from);
        if (result.find(v_set_from) == result.end()) {
            result.emplace(std::make_pair(
                v_set_from, std::vector<int>{v_set_to})
            );
        } else {
            result[v_set_from].push_back(v_set_to);
        }
    }
    return result;
}

std::unordered_map<int, std::vector<int>> expand_arborescence(
    const std::vector<float>& weights,
    const std::unordered_map<int, std::vector<int>>& contracted_arborescence,
    const DisjointSetUnion& sets,
    const DisjointSetUnion& new_sets,
    std::unordered_map<int, int>& min_x,
    std::unordered_map<int, int>& min_y,
    std::unordered_map<int, std::vector<int>>& v_in_sets,
    std::unordered_map<int, std::vector<int>>& new_v_in_sets,
    const std::vector<int>& cycle,
    int v_cycle,
    int max_n,
    int batch_idx,
    int length)
{
    std::unordered_map<int, std::vector<int>> result;

    int v_to_delete;
    int pi_v_to_delete;
    for (auto iter = contracted_arborescence.begin(); iter != contracted_arborescence.end(); ++iter) {
        int from = iter->first;
        for (int to : iter->second) {
            if (from != v_cycle && to != v_cycle) {
                if (result.find(from) == result.end()) {
                    result.emplace(std::make_pair(from, std::vector<int>{to}));
                } else {
                    result[from].push_back(to);
                }
            } else if (to == v_cycle) {
                // given (u, v_c) find corresponding (u, v), v = argmin{w(u, v)|v in v_c }
                float min_value = -1;
                int u_res;
                int v_res;
                for (int u_original : v_in_sets[from]) {
                    for (int v_original : new_v_in_sets[v_cycle]) {
                        float cur_value = weights[idx(batch_idx, u_original, v_original, max_n)];
                        if (min_value == -1 || cur_value < min_value) {
                            min_value = cur_value;
                            u_res = u_original;
                            v_res = v_original;
                        }
                    }
                }

                v_to_delete = sets.find(v_res);
                pi_v_to_delete = min_x[v_to_delete];
                pi_v_to_delete = sets.find(pi_v_to_delete);

                int u_to_append = sets.find(u_res);
                if (result.find(u_to_append) == result.end()) {
                    result.emplace(std::make_pair(u_to_append, std::vector<int>{v_to_delete}));
                } else {
                    result[u_to_append].push_back(v_to_delete);
                }

            }
        }
    }

    for (int i = 0; i != cycle.size(); ++i) {
        int to = cycle[i];
        int from = cycle[(i + 1) % cycle.size()];

        if (from != pi_v_to_delete || to != v_to_delete) {
            if (result.find(from) == result.end()) {
                result.emplace(std::make_pair(from, std::vector<int>{to}));
            } else {
                result[from].push_back(to);
            }
        }
    }

    for (auto iter = contracted_arborescence.begin(); iter != contracted_arborescence.end(); ++iter) {
        int from = iter->first;
        for (int to : iter->second) {
            if (from == v_cycle) {
                float min_value = -1;
                int u_res;
                int v_res;
                for (int u_original : new_v_in_sets[v_cycle]) {
                    for (int v_original : v_in_sets[to]) {
                        float cur_value = weights[idx(batch_idx, u_original, v_original, max_n)];
                        if (min_value == -1 || cur_value < min_value) {
                            min_value = cur_value;
                            u_res = u_original;
                            v_res = v_original;
                        }
                    }
                }
                int u_set = sets.find(u_res);
                int v_set = sets.find(v_res);
                if (result.find(u_set) == result.end()) {
                    result.emplace(std::make_pair(u_set, std::vector<int>{v_set}));
                } else {
                    result[u_set].push_back(v_set);
                }
            }
        }
    }

    return result;
}

std::vector<std::vector<int>> adj_matrix(int n, const std::unordered_map<int, std::vector<int>>& graph) {
    std::vector<std::vector<int>> res(n,
        std::vector<int>(n));

    for (auto iter = graph.begin(); iter != graph.end(); ++iter) {
        int x = iter->first;
        for (int y : iter->second) {
            res[x][y] = 1;
        }
    }

    return res;
}

std::vector<int> lin_adj_matrix(int n, const std::unordered_map<int, std::vector<int>>& graph) {
    std::vector<int> res(n * n, 0);
    for (auto iter = graph.begin(); iter != graph.end(); ++iter) {
        int x = iter->first;
        for (int y : iter->second) {
            int index = x * n + y;
            res[index] = 1;
        }
    }

    return res;
}

void update_stats(
    MinStats& stats,
    const std::vector<float>& weights,
    std::unordered_map<int, int>& min_x,
    std::unordered_map<int, int>& min_y,
    std::unordered_map<int, std::vector<int>>& v_in_sets,
    int max_n,
    int batch_idx,
    int length)
{
    for (auto iter = min_x.begin(); iter != min_x.end(); ++iter) {
        int key = iter->first;
        int cur_x = iter->second;
        int cur_y = min_y[key];

        if (weights[idx(batch_idx, cur_x, cur_y, max_n)] != 0) {
            std::vector<int> mask(length);
            for (int v_original : v_in_sets[key]) {
                mask[v_original] = 1;
            }

            stats.min_x.push_back(cur_x);
            stats.min_y.push_back(cur_y);
            stats.mask.push_back(mask);
        }
    }
}

std::unordered_map<int, std::vector<int>> get_arborescence(
    std::vector<float>& weights,
    int root,
    DisjointSetUnion& sets,
    MinStats& stats,
    int batch_size,
    int max_n,
    int batch_idx,
    int length)
{
    std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>> mins = get_minima(
        weights, sets, root, max_n, batch_idx, length);

    std::unordered_map<int, int> min_x = mins.first;
    std::unordered_map<int, int> min_y = mins.second;

    std::unordered_map<int, std::vector<int>> v_in_sets = sets.get_sets();

    update_stats(stats, weights, min_x, min_y, v_in_sets, max_n, batch_idx, length);

    std::unordered_map<int, std::vector<int>> edges = get_min_graph(min_x, sets);
    std::vector<int> cycle = find_cycle(edges);

    if (cycle.size() == 0) {
        return edges;
    }

    DisjointSetUnion new_sets(sets);
    new_sets.unite_many(cycle);

    int v_cycle = new_sets.find(cycle[0]);

    for (int j = 0; j != length; ++j) {
        int set_j = sets.find(j);
        if (set_j == sets.find(root)) {
            continue;
        }

        float min_value = weights[idx(batch_idx, min_x[set_j], min_y[set_j], max_n)];
        for (int i = 0; i != length; ++i) {
            weights[idx(batch_idx, i, j, max_n)] -= min_value;
        }
    }

    std::unordered_map<int, std::vector<int>> contracted_arborescence = get_arborescence(
        weights,
        new_sets.find(root),
        new_sets,
        stats,
        batch_size,
        max_n,
        batch_idx,
        length
    );

    std::unordered_map<int, std::vector<int>> new_v_in_sets = new_sets.get_sets();

    return expand_arborescence(
        weights,
        contracted_arborescence,
        sets,
        new_sets,
        min_x,
        min_y,
        v_in_sets,
        new_v_in_sets,
        cycle,
        v_cycle,
        max_n,
        batch_idx,
        length
    );
}

template <typename T>
std::vector<T> linearize(const std::vector<std::vector<T>>& mtx) {
    std::vector<T> result;
    result.reserve(mtx.size() * mtx[0].size());
    for (int i = 0; i != mtx.size(); ++i) {
        for (int j = 0; j != mtx[i].size(); ++j) {
            result.push_back(mtx[i][j]);
        }
    }

    return result;
}

using ArbStats = std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>;

ArbStats convert_to_tensors(
    std::vector<int>& lin_arb,
    std::vector<MinStats>& min_stats,
    std::vector<int>& lengths,
    int batch_size,
    int max_n)
{
    auto opts = torch::TensorOptions().dtype(torch::kInt32);

    torch::Tensor arb_res = torch::from_blob(lin_arb.data(), {batch_size, max_n, max_n}, opts).to(torch::kInt64);

    std::vector<torch::Tensor> min_xs;
    std::vector<torch::Tensor> min_ys;
    std::vector<torch::Tensor> masks;

    min_xs.reserve(batch_size);
    min_ys.reserve(batch_size);
    masks.reserve(batch_size);

    for (int i = 0; i != batch_size; ++i) {
        torch::Tensor min_x = torch::from_blob(min_stats[i].min_x.data(), {min_stats[i].min_x.size()}, opts).to(torch::kInt64);
        torch::Tensor min_y = torch::from_blob(min_stats[i].min_y.data(), {min_stats[i].min_y.size()}, opts).to(torch::kInt64);
        std::vector<int> mask = linearize(min_stats[i].mask);
        torch::Tensor mask_res = torch::from_blob(mask.data(), {min_stats[i].mask.size(), lengths[i]}, opts).to(torch::kInt64);
        min_xs.push_back(min_x);
        min_ys.push_back(min_y);
        masks.push_back(mask_res);
    }

    return std::make_tuple(arb_res, min_xs, min_ys, masks);
}

ArbStats get_arborescence_batch(torch::Tensor weights, int root, torch::Tensor lengths) {
    int batch_size = weights.size(0);
    int max_n = weights.size(1);
    std::vector<float> weights_vec(weights.data_ptr<float>(), weights.data_ptr<float>() + weights.numel());
    std::vector<int> lengths_vec(lengths.data_ptr<int>(), lengths.data_ptr<int>() + lengths.numel());

    std::vector<int> arb_res(batch_size * max_n * max_n);
    std::vector<MinStats> min_stats(batch_size);

    for (int i = 0; i != batch_size; ++i) {
        int cur_length = lengths_vec[i];
        DisjointSetUnion sets(cur_length);
        MinStats stats;
        std::unordered_map<int, std::vector<int>> arb = get_arborescence(
            weights_vec,
            root,
            sets,
            stats,
            batch_size,
            max_n,
            i,
            cur_length
        );

        for (auto iter = arb.begin(); iter != arb.end(); ++iter) {
            int x = iter->first;
            for (int y : iter->second) {
                arb_res[idx(i, x, y, max_n)] = 1;
            }
        }

        min_stats[i] = stats;
    }

    return convert_to_tensors(arb_res, min_stats, lengths_vec, batch_size, max_n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_arborescence_batch",
        &get_arborescence_batch,
        "Get minimum spanning arborescences over batch");
}
