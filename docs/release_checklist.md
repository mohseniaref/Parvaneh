# Independent release checklist

1. Confirm author name, ORCID, repository URL, version, and release date in
   `CITATION.cff` and `.zenodo.json`.
2. Run `pytest -q` from a fresh environment with no external reference files.
3. Execute the synthetic notebook and both benchmark programs.
4. Confirm every committed notebook holds the outputs of its last successful
   run: execution counts `1`, `2`, `3`, ... in cell order, and no stored
   traceback.
5. Build the allowlisted archive with `python tools/build_zenodo_archive.py`.
6. Inspect `ARCHIVE_MANIFEST.json` and verify that the archive contains only
   package code, tests, synthetic examples, and original documentation.
7. Upload the ZIP and SHA-256 checksum to Zenodo, reserve the DOI, update
   citation metadata, rebuild, and then publish the final archive.

The public release must remain executable without a book PDF, historical
source tree, external executable, or supplied example raster.
