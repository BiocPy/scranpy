#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "mumosa/mumosa.hpp"

#include "utils.h"

pybind11::array scale_by_neighbors(const pybind11::list& distances) {
    size_t nmod = distances.size();
    std::vector<std::pair<double, double> > values;
    values.reserve(nmod);

    for (size_t x = 0; x < nmod; ++x) {
        const auto& dist = distances[x].cast<pybind11::array>();
        auto dptr = check_numpy_array<double>(dist);
        std::vector<double> copy(dptr, dptr + dist.size()); // creating a copy as compute_distance() will shuffle the distances to compute the median.
        values.push_back(mumosa::compute_distance(copy.size(), copy.data()));
    }

    auto output = mumosa::compute_scale<double>(values);
    return pybind11::array_t<double>(output.size(), output.data());
}

void init_scale_by_neighbors(pybind11::module& m) {
    m.def("scale_by_neighbors", &scale_by_neighbors);
}
