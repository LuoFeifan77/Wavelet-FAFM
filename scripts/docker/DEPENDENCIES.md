# Restored OpenPI build files

The FAFM anonymous repository at https://anonymous.4open.science/r/FAFM/
does not publish `pyproject.toml`, `uv.lock`, or `packages/openpi-client`.
Its README refers installation to OpenPI.

These files, together with the upstream `LICENSE`, were restored from:

https://github.com/Physical-Intelligence/openpi/tree/2c258f104d4c9b63e430410de5be3168ae510517

The existing `scripts/docker/serve_policy.Dockerfile` and every file in
`src/openpi/models_pytorch/` match this upstream commit byte for byte. This
provides a matching dependency baseline, but does not establish the exact
environment used by the FAFM authors. The restored manifest, lockfile, and
client package are unchanged from that commit. The license text is preserved
with a final newline added. FAFM model and training code has not been replaced.

The lockfile pins Transformers 4.53.2, JAX 0.5.3, Flax 0.10.2, and the
LeRobot Git revision used by that OpenPI version. Keep the root manifest,
lockfile, and client package together when moving the project.

From the project root, build and start the server with:

```bash
docker compose -f scripts/docker/compose.yml up --build
```

The upstream `LICENSE` is retained for the restored upstream material; this
restoration does not establish licensing for FAFM-specific additions.

FAFM training scripts additionally import `swanlab`, which is absent from
the upstream dependency manifest. Training needs that dependency separately;
this restoration targets the existing Docker policy-serving configuration.
