# ViTALBench GitHub Repository Cleanup

## Goal

Clean the public repository structure and documentation without publishing agent-related files.

## Repository layout

Move the four question dataset files into `data/questions/`:

- `detection.json`
- `detection_e.json`
- `law.json`
- `law_e.json`

Move the legal metadata and corpus files into `data/legal/`:

- `law_categories.json`
- `law_db.json`

Update all scripts, tests, notebooks, and documentation that reference the old paths.

## Excluded content

The repository must not track `AGENTS.md` files, `.idea/`, or `backup/` directories.

The `.gitignore` file will exclude these paths and other local or generated artifacts without ignoring the published Fig. 3 notebook.

## Documentation

Update `README.md` with the new data layout, canonical file paths, audit commands, and reproducibility instructions.

The README will not mention the agent pipeline.

## Validation

Run the release statistics audit and the test suite.

Search tracked files for forbidden agent-related paths and confirm none are present.

Check that the working tree contains only intended changes before committing.

## Scope

This cleanup changes file organization, path references, ignore rules, and README documentation only.

It does not modify dataset contents, paper prose, images, model behavior, or files under `data/` other than moving the specified JSON files.
