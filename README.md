# geolifeclef-2024

## Quickstart

Install the pre-commit hooks for formatting code:

```bash
pre-commit install
```

We are generally using a shared VM with limited space.
Install packages to the system using sudo:

```bash
sudo pip install -r requirements.txt
```

We can ignore the following message since we know what we are doing:

```
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
```

This should allow all users to use the packages without having to install them multiple times.
This is a problem with very large packages like `torch` and `spark`.

Then install the current package into your local user directory, so changes that you make are local to your own users.

```bash
pip install -e .
```

## Notes

### Processing tiles

On a VM with at least 700GB of disk (or more), we can process tiles into parquet for further processing.
We might want to do this because we generally iterate through tiles in batches of 100 at 3-4 iterations per second.
It makes sense to then write out the tiles to disk and pre-process them for downstream processing e.g. tile2vec-styled sampling.

```bash
# first sync data from the bucket locally
python -m geolifeclef.workflows.sync_data

# then run the _very_ slow process of generating the tiles
python -m geolifeclef.workflows.tiles_parquet
```
