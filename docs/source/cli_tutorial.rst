.. _sec_cli_tutorial:

CLI Tutorial
============

This tutorial walks through the same instance segmentation pipeline as the
:ref:`Python tutorial <sec_tutorial>`, but driven entirely from the command line
using ``volara-cli``.

Prerequisites
-------------

Install volara with its docs extras (needed for the data preparation step)::

    uv pip install volara[docs]

Set up a working directory and copy the helper scripts into it::

    mkdir volara-tutorial && cd volara-tutorial
    cp /path/to/volara/docs/cli_tut_scripts/*.py .

.. note::

   Adjust the ``cp`` path above to wherever you cloned the volara repository.

Step 1: Prepare the Toy Data
-----------------------------

The tutorial uses a small 2-channel fluorescence microscopy volume (nuclei and
cell membranes) from scikit-image. Run the preparation script to create a zarr
container with raw data, labels, and perfect affinities::

    python prepare_data.py

This creates ``cells3d.zarr/`` with four datasets: ``raw``, ``mask``,
``labels``, and ``affs``.

Step 2: View the Data
---------------------

Use ``show_slice.py`` to inspect slices of the data at Z=30::

    python show_slice.py cells3d.zarr/raw -o raw.png
    python show_slice.py cells3d.zarr/labels -o labels.png
    python show_slice.py cells3d.zarr/affs -o affs.png

Open the resulting PNG files in your image viewer to see:

- **raw.png** — the 2-channel fluorescence image (nuclei in green, membranes in
  magenta)
- **labels.png** — pseudo ground truth from connected components
- **affs.png** — perfect affinities derived from labels

Step 3: Extract Fragments
--------------------------

Now we run the first volara task: watershed-based supervoxel extraction.
All parameters are passed directly on the command line. Use dotted keys
(like ``--db.path``) for nested fields, JSON strings for complex values,
and space-separated numbers for lists::

    volara-cli run extract-frags \
        --db.db_type sqlite \
        --db.path cells3d.zarr/db.sqlite \
        --db.edge_attrs '{"zyx_aff":"float"}' \
        --affs_data.store cells3d.zarr/affs \
        --affs_data.neighborhood '[[1,0,0],[0,1,0],[0,0,1]]' \
        --frags_data cells3d.zarr/fragments \
        --block_size 20 100 100 \
        --context 2 2 2 \
        --bias -0.5 -0.5 -0.5

A few things to note:

- **Dotted keys** like ``--db.path`` build nested structures:
  ``--db.db_type sqlite --db.path foo.sqlite`` becomes
  ``{"db": {"db_type": "sqlite", "path": "foo.sqlite"}}``.
- **Multi-value arguments** like ``--block_size 20 100 100`` are collected
  into lists automatically.
- **Negative numbers** like ``--bias -0.5 -0.5 -0.5`` are handled correctly
  (they aren't mistaken for flags).
- **JSON strings** like ``--db.edge_attrs '{"zyx_aff":"float"}'`` are parsed
  for complex nested values.
- **Shorthands:** ``--frags_data cells3d.zarr/fragments`` is shorthand for
  ``--frags_data.store cells3d.zarr/fragments``. When a field expects a
  Pydantic model and you pass a bare string, it is assigned to that model's
  first required field.

**What's happening:** The volume is divided into overlapping blocks
(20x100x100 voxels plus 2 voxels of context on each side). Within each block,
a mutex watershed runs on the affinities to produce supervoxels (fragments).
The ``bias`` of -0.5 shifts affinities from the range (0, 1) to (-0.5, 0.5),
making boundaries split and interiors merge.

View the fragments::

    python show_slice.py cells3d.zarr/fragments -o fragments.png

You should see colored supervoxels with visible block boundaries — this is
expected for a blockwise approach.

Step 4: Compute Edges
---------------------

Next, compute affinity-based edges between neighboring fragments::

    volara-cli run aff-agglom \
        --db.db_type sqlite \
        --db.path cells3d.zarr/db.sqlite \
        --db.edge_attrs '{"zyx_aff":"float"}' \
        --affs_data.store cells3d.zarr/affs \
        --affs_data.neighborhood '[[1,0,0],[0,1,0],[0,0,1]]' \
        --frags_data cells3d.zarr/fragments \
        --block_size 20 100 100 \
        --context 2 2 2 \
        --scores '{"zyx_aff":[[1,0,0],[0,1,0],[0,0,1]]}'

This populates the fragment graph in the SQLite database with edges
weighted by affinity scores. These edges tell us how confident we are
that neighboring fragments belong to the same object.

Step 5: Global Matching
-----------------------

Now we solve the graph globally using mutex watershed to produce a
fragment-to-segment lookup table. This step reads the entire graph into
memory (just nodes and edges, not voxels) and finds the optimal
segmentation.

Notice that ``--db`` and ``--lut`` use shorthands here — passing a bare string
for a field that expects a Pydantic model automatically assigns it to that
model's first required field. So ``--db cells3d.zarr/db.sqlite`` expands to
``--db.db_type sqlite --db.path cells3d.zarr/db.sqlite``. We don't need to
specify ``edge_attrs`` because the database already exists and volara reads
the schema from it::

    volara-cli run graph-mws \
        --db cells3d.zarr/db.sqlite \
        --lut cells3d.zarr/lut \
        --weights '{"zyx_aff":[1.0,-0.5]}' \
        --roi '[[0,0,0],[17400,66560,66560]]'

.. note::

   The ``roi`` is specified in world coordinates (nanometers) as
   ``[offset, shape]``. For our data: 60 Z-slices at 290 nm = 17400 nm,
   256 pixels at 260 nm = 66560 nm.

The output is a lookup table (``cells3d.zarr/lut.npz``) mapping each
fragment ID to a segment ID.

Step 6: Relabel Fragments
--------------------------

Apply the lookup table to produce the final segmentation volume. This
command is short enough that shorthands and simple args are all you need::

    volara-cli run relabel \
        --frags_data cells3d.zarr/fragments \
        --seg_data cells3d.zarr/segments \
        --lut cells3d.zarr/lut \
        --block_size 20 100 100

Step 7: View and Evaluate
--------------------------

View the final segmentation::

    python show_slice.py cells3d.zarr/segments -o segments.png

Compare to ``labels.png`` — the segments should closely match the pseudo
ground truth.

Run the evaluation script::

    python evaluate.py

With perfect affinities, you should see 100% accuracy (zero false merges
and zero false splits).

Using Config Files
------------------

The commands above work well, but for complex or repeated configurations,
a YAML config file is more convenient. For example, create
``extract_frags.yaml``::

    cat > extract_frags.yaml << 'EOF'
    db:
      db_type: sqlite
      path: cells3d.zarr/db.sqlite
      edge_attrs:
        zyx_aff: float

    affs_data:
      store: cells3d.zarr/affs
      neighborhood:
        - [1, 0, 0]
        - [0, 1, 0]
        - [0, 0, 1]

    frags_data:
      store: cells3d.zarr/fragments

    block_size: [20, 100, 100]
    context: [2, 2, 2]
    bias: [-0.5, -0.5, -0.5]
    EOF

Then run with ``-c``::

    volara-cli run extract-frags -c extract_frags.yaml

Mixing Config Files and CLI Arguments
--------------------------------------

CLI arguments and config files can be combined — CLI arguments override
config file values. This is useful for experimenting with parameters
without editing the file each time.

Re-run fragment extraction with a different bias::

    volara-cli run extract-frags -c extract_frags.yaml --bias -0.3 -0.3 -0.3

Override the block size::

    volara-cli run extract-frags -c extract_frags.yaml --block_size 30 128 128

Override a nested field::

    volara-cli run extract-frags -c extract_frags.yaml --db.path other.sqlite

Discovering Available Tasks
----------------------------

To see all available task types::

    volara-cli run --help

The help output lists every registered task name that can be passed as the
first argument to ``volara-cli run``.

Cleaning Up
-----------

Remove the generated data::

    rm -rf cells3d.zarr volara_logs
