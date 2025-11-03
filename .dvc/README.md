This folder stores DVC configuration for data versioning and pipelines.

- `config`: repository-visible DVC config (remote **URL only**, no secrets).
- `config.local.example`: template for local, **untracked** credentials (copy to `.dvc/config.local`).
- `.gitignore`: ensures `.dvc/config.local` is not committed.

## How to configure locally

1. Copy the example:
cp .dvc/config.local.example .dvc/config.local

2. Fill your Google Drive credentials (client_id / client_secret) or use a service account.

3. Test the connection:
```
dvc pull
dvc repro
pgsql
```

**Important:** Never commit `.dvc/config.local`. Secrets must remain local.
