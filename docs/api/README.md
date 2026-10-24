# API Documentation

This directory is for generated/reference documentation of the
FastAPI inference service defined in `serving/app/`.

## Interactive docs

FastAPI generates interactive API documentation automatically from
`serving/app/schemas.py` and the route definitions in
`serving/app/routes/`. With the service running (see the root
`README.md`'s "Running the inference service" section), it's
available at:

- `http://localhost:8000/docs` (Swagger UI)
- `http://localhost:8000/redoc` (ReDoc)

Prefer linking to a running instance over hand-maintaining a static
copy here - a hand-written API doc drifts out of sync with the actual
schema the moment someone edits `serving/app/schemas.py` without
remembering to update a second copy.

## What belongs in this directory

Static reference material that the interactive docs above don't
cover: authentication/deployment notes for API consumers, a changelog
of breaking schema changes between versions, or an exported OpenAPI
spec (`openapi.json`) snapshotted at a release tag for external
integrators who need something they can diff.
