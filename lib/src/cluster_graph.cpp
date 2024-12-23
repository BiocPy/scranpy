#include <vector>
#include <stdexcept>
#include <memory>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "scran_graph_cluster/scran_graph_cluster.hpp"

#include "utils.h"

static std::pair<raiigraph::Graph, std::unique_ptr<igraph_vector_t> > formulate_graph(const pybind11::tuple& graph) {
    if (graph.size() != 3) {
        throw std::runtime_error("graph should be represented by a 3-tuple");
    }
    size_t vertices = graph[0].cast<size_t>();
    const auto& edges = graph[1].cast<pybind11::array>();

    std::unique_ptr<igraph_vector_t> weight_view_ptr;
    if (!pybind11::isinstance<pybind11::none>(graph[2])) {
        const auto& weights = graph[2].cast<pybind11::array>();
        weight_view_ptr = std::make_unique<igraph_vector_t>();
        igraph_vector_view(weight_view_ptr.get(), check_numpy_array<igraph_real_t>(weights), weights.size());
    }

    return std::make_pair(
        scran_graph_cluster::edges_to_graph(edges.size(), check_numpy_array<igraph_integer_t>(edges), vertices, false),
        std::move(weight_view_ptr)
    );
}

pybind11::tuple cluster_multilevel(const pybind11::tuple& graph, double resolution, int seed) {
    auto gpair = formulate_graph(graph);

    scran_graph_cluster::ClusterMultilevelOptions opt;
    opt.resolution = resolution;
    opt.seed = seed;
    scran_graph_cluster::ClusterMultilevelResults res;
    scran_graph_cluster::cluster_multilevel(gpair.first.get(), gpair.second.get(), opt, res);

    size_t nlevels = res.levels.nrow();
    pybind11::tuple levels(nlevels);
    for (size_t l = 0; l < nlevels; ++l) {
        auto incol = res.levels.row(l);
        pybind11::array_t<igraph_integer_t> current(incol.size());
        std::copy(incol.begin(), incol.end(), static_cast<igraph_integer_t*>(current.request().ptr));
        levels[l] = std::move(current);
    }

    pybind11::tuple output(4);
    output[0] = res.status;
    output[1] = pybind11::array_t<igraph_integer_t>(res.membership.size(), res.membership.data());
    output[2] = levels;
    output[3] = pybind11::array_t<igraph_real_t>(res.modularity.size(), res.modularity.data());

    return output;
}

pybind11::tuple cluster_leiden(const pybind11::tuple& graph, double resolution, bool use_cpm, int seed) {
    auto gpair = formulate_graph(graph);

    scran_graph_cluster::ClusterLeidenOptions opt;
    opt.resolution = resolution;
    opt.modularity = !use_cpm;
    opt.seed = seed;
    opt.report_quality = true;
    scran_graph_cluster::ClusterLeidenResults res;
    scran_graph_cluster::cluster_leiden(gpair.first.get(), gpair.second.get(), opt, res);

    pybind11::tuple output(3);
    output[0] = res.status;
    output[1] = pybind11::array_t<igraph_integer_t>(res.membership.size(), res.membership.data());
    output[2] = res.quality;

    return output;
}

pybind11::tuple cluster_walktrap(const pybind11::tuple& graph, int steps) {
    auto gpair = formulate_graph(graph);

    scran_graph_cluster::ClusterWalktrapOptions opt;
    opt.steps = steps;
    scran_graph_cluster::ClusterWalktrapResults res;
    scran_graph_cluster::cluster_walktrap(gpair.first.get(), gpair.second.get(), opt, res);

    size_t merge_nrow = res.merges.nrow(), merge_ncol = res.merges.ncol();
    pybind11::array_t<igraph_integer_t, pybind11::array::f_style> merges({ merge_nrow, merge_ncol });
    for (size_t m = 0; m < merge_ncol; ++m) {
        auto incol = res.merges.column(m);
        auto outptr = static_cast<igraph_integer_t*>(merges.request().ptr) + m * merge_nrow;
        std::copy(incol.begin(), incol.end(), outptr);
    }

    pybind11::tuple output(4);
    output[0] = res.status;
    output[1] = pybind11::array_t<igraph_integer_t>(res.membership.size(), res.membership.data());
    output[2] = merges;
    output[3] = pybind11::array_t<igraph_real_t>(res.modularity.size(), res.modularity.data());

    return output;
}

void init_cluster_graph(pybind11::module& m) {
    m.def("cluster_multilevel", &cluster_multilevel);
    m.def("cluster_leiden", &cluster_leiden);
    m.def("cluster_walktrap", &cluster_walktrap);
}
