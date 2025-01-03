# library(testthat); library(scrapper); source("test-clusterKmeans.R")
import scranpy
import numpy


def test_cluster_kmeans_basic():
    x = numpy.random.randn(20, 1000)
    clustering = scranpy.cluster_kmeans(x, 10)
    assert len(clustering.clusters) == 1000
    assert len(set(clustering.clusters)) == 10
    assert clustering.centers.shape == (20, 10)

    # Randomness should be fully controlled.
    again = scranpy.cluster_kmeans(x, 10)
    assert (again.clusters == clustering.clusters).all()
    assert (again.centers == clustering.centers).all()


def test_cluster_kmeans_alt():
    x = numpy.random.randn(20, 1000)
    clustering = scranpy.cluster_kmeans(x, 8, init_method="random")
    assert len(clustering.clusters) == 1000
    assert len(set(clustering.clusters)) == 8 
    assert clustering.centers.shape == (20, 8)

    clustering = scranpy.cluster_kmeans(x, 5, init_method="kmeans++", refine_method="lloyd")
    assert len(clustering.clusters) == 1000
    assert len(set(clustering.clusters)) == 5 
    assert clustering.centers.shape == (20, 5)
