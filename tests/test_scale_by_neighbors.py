import scranpy
import numpy
import knncolle


def test_scale_by_neighbors():
    numpy.random.seed(200)
    embeddings = [
        numpy.random.randn(50, 200), 
        numpy.random.randn(5, 200) * 3, 
        numpy.random.randn(10, 200) * 5
    ]

    out = scranpy.scale_by_neighbors(embeddings, nn_parameters=knncolle.VptreeParameters())
    assert len(out.scaling) == 3
    assert out.combined.shape == (65, 200)

    # Comparing it to a reference calculation.
    manual_knn_dist = []
    k = 20
    for embed in embeddings:
        distances = []
        for i in range(embed.shape[1]):
            chosen = embed[:,i]
            curdist = numpy.sqrt(((embed.T - chosen)**2).sum(axis=1))
            distances.append(sorted(curdist)[k]) # zero-indexed, so should be 'k - 1'; but we use 'k' to skip the distance of zero to itself.
        manual_knn_dist.append(numpy.median(distances))
    assert numpy.allclose(out.scaling, manual_knn_dist[0]/manual_knn_dist) # first embedding is the reference.
