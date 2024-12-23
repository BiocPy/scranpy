#!/bin/bash

set -e
set -u

IGRAPH_VERSION=0.10.15
IGRAPH_HASH=03ba01db0544c4e32e51ab66f2356a034394533f61b4e14d769b9bbf5ad5e52c
SOURCE_DIR=igraph-${IGRAPH_VERSION}

if [[ ! -e ${SOURCE_DIR} ]]
then
    wget -q https://github.com/igraph/igraph/releases/download/${IGRAPH_VERSION}/igraph-${IGRAPH_VERSION}.tar.gz -O igraph.tar.gz
    OBSERVED_HASH=($(shasum -a 256 igraph.tar.gz))
    if [[ ${OBSERVED_HASH} != ${IGRAPH_HASH} ]]
    then
        echo "hash mismatch for ${IGRAPH_VERSION} (got ${OBSERVED_HASH})"
        exit 1
    fi
    tar -xf igraph.tar.gz
fi

BUILD_DIR=build-${IGRAPH_VERSION}
if [ ! -e ${BUILD_DIR} ]
then
    mkdir -p ../installed
    cmake \
        -S ${SOURCE_DIR} \
        -B ${BUILD_DIR} \
        -DCMAKE_POSITION_INDEPENDENT_CODE=true \
        -DIGRAPH_WARNINGS_AS_ERRORS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$(pwd)/../installed
fi

cmake --build ${BUILD_DIR}
cmake --install ${BUILD_DIR}
