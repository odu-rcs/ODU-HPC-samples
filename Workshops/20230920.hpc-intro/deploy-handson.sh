#!/bin/bash

set -e
set -u

WORKSHOP_DEST_DIR=/cm/shared/workshops/intro-to-hpc/20230920.hpc-intro

# SRC_DIR is the ODU-HPC-samples repo root, from which we will source
# all the workshop hands-on files (FIXME make this movable)
SRC_DIR=/home/wpurwant/ODU-HPC-samples

WORKSHOP_SRC_DIR="$SRC_DIR/Workshops/20230920.hpc-intro"
SRC_LIST_FILE="$SRC_DIR/Workshops/20230920.hpc-intro/handson-files.list"


# _sync_orig_files: Bulk copying of the original files
_sync_orig_files () {
    rsync -a -v --update "$SRC_DIR/."  "$WORKSHOP_DEST_DIR/." \
          --files-from="$SRC_LIST_FILE"
}

_deploy_READMEs () {
    rsync -a -v --update "$WORKSHOP_SRC_DIR/README.md"  "$WORKSHOP_DEST_DIR/."
}


# _reorg_python_pi: Reorganizes files in the `python/pi-serial`
# and `python/pi-parallel` folders for hands-on.
_reorg_python_pi () {
    local ORIG_PWD
    ORIG_PWD="$PWD"
    cd "$WORKSHOP_DEST_DIR"
    mkdir -p python/pi-serial/solutions
    mv  python/pi-serial/*.slurm  python/pi-serial/solutions

    mkdir -p python/pi-parallel/solutions
    mv  python/pi-parallel/*.slurm  python/pi-parallel/solutions
    cd "$ORIG_PWD"
}


main1 () {
    set -x
    _sync_orig_files
    _deploy_READMEs
    _reorg_python_pi
}


main1 "$@"
