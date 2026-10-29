# Troubleshooting

Problems met while setting up FICL, and their fixes. Use `python -m pip` rather than a bare `pip`,
so that pip and Python always belong to the same environment.

| Symptom | Cause | Fix |
|---|---|---|
| `UserWarning: Failed to initialize NumPy: _ARRAY_API not found`; `RuntimeError: Could not infer dtype of numpy.int64` or `Numpy is not available`; `test_torch_and_numpy_exchange_arrays` fails | NumPy 2 next to PyTorch 2.0.1, which is built against NumPy 1.x; usually pulled in by a pip dependency | `python -m pip install "numpy<2"`, then `python -m pip check`. If that lists a package that needs NumPy 2, rebuild the environment: `conda deactivate`, `conda env remove -n ficl`, `conda env create -f env_cuda.yaml` |
| `pip check`: `pandas 3.x has requirement numpy>=2.0.2` | pandas comes in through medmnist, and pandas 3.x needs NumPy 2 | `python -m pip install "numpy<2" "pandas<3"` (`env_cuda.yaml` pins both for new environments) |
| `pip check`: `cvxpy ... has requirement numpy>=2.0.0` (or `sparsediffpy`) | the environment was cloned from an HtFLlib environment; FICL does not use cvxpy | rebuild the environment from `env_cuda.yaml`, or `python -m pip uninstall cvxpy sparsediffpy` and check again |
| pip reports NumPy 1.26, but `python -c "import numpy; print(numpy.__version__)"` prints 2.x | Python imports a second NumPy: from the user site `~/.local/lib/python3.11/site-packages` (searched before the environment), from a folder on `PYTHONPATH`, or from inside the environment next to the one pip manages | `python -c "import numpy; print(numpy.__file__)"` shows which one. User site: `conda env config vars set PYTHONNOUSERSITE=1 -n ficl`, then reactivate. `PYTHONPATH`: remove the folder (for example in `~/.bashrc`). Inside the environment: rebuild it |
| `Permission denied` when starting `run_tests.sh`, `system/run.sh` or `system/smoke.sh` | the files lost their executable bit when copied | `chmod +x run_tests.sh system/run.sh system/smoke.sh`, or start them with `bash` |
| `Scenario not found` | the scenario of that seed was never generated | the error prints the `generate_scenario.py` command that makes it; generate one folder per seed with `--num_seeds` |
| `... has no automatic download` | RESISC45 is a manual download | download `NWPU-RESISC45.rar`, extract it into `dataset/RESISC45/rawdata/`, run `python prepare_data.py RESISC45` |
| `could not download ...: HTTP Error 403`, `ContentTooShortError` or `retrieval incomplete` | the host refused urllib's default client or the connection was cut; `prepare_data.py` now sends a browser-like User-Agent and resumes cut downloads (5 tries) | download the archive yourself into `<Name>/rawdata/` with the exact file name (`wget -c -O <Name>/rawdata/<file> "<url>"`, URLs in `utils/raw_utils.py`, `BENCHMARKS`) and run `prepare_data.py <Name> --download` again: an archive already there is only unpacked |
| `WARNING: ... has md5 ...` after a download | the archive differs from the published one, usually a download that was cut off | delete the archive in `rawdata/` and run `prepare_data.py ... --download` again |
| `... was written before the synthetic image settings were recorded` | `dataset/Synthetic` was made by an earlier version | `cd dataset && python make_synthetic.py`: it keeps the index (same sha256), the caches and the scenarios, and adds the settings to `index.json` |
| `the raw files of <Name> no longer give the rows of ...cl_index` | images were added, removed or renamed in `<Name>/rawdata/` after the index was built, so a row would read another image | restore the files; or rebuild the index with `prepare_data.py <Name> --force` and then every scenario and cache built on it |
| `RuntimeError: Hugging Face hub model specified but package not installed` (timm, `vit_b16_in21k`), or FICL's `timm 0.9.16 cannot use huggingface_hub` | huggingface_hub is missing, or it is 2.x, which no longer exports a name timm 0.9.16 imports (timm then reports it as not installed) | `python -m pip install "huggingface_hub<1"`, then `python -m pip check` (`env_cuda.yaml` pins it for new environments) |
| `... trains a network and needs torch` | an image algorithm ran in an environment without PyTorch | activate the environment of `env_cuda.yaml` |
| the run stops while loading `vit_b16_in21k`, `dinov2_vitb14` or `clip_vitb16` (connection errors) | the pretrained weights download on first use: timm from Hugging Face, DINOv2 through torch.hub (code from GitHub), CLIP from OpenAI's URL | run once with internet access, or copy the caches from a machine that has them: `~/.cache/huggingface`, `~/.cache/torch/hub`, `~/.cache/clip` |
| `CUDA out of memory` in an image run | batch, model or evaluation batch too large for the GPU | lower `-lbs` (training) or `-ebs` (evaluation); `full` fine-tuning of a ViT-B needs the most memory, `head`, `lora` and `prompt` the least |
| `DataLoader worker ... exited unexpectedly` or `bus error` in an image run | the loader workers share images through `/dev/shm`, which is small in some containers | `-nw 0` (decoding in the main process; results do not change), or give the container more shared memory |
| `UserWarning: ... does not have a deterministic implementation` | an operation of the network has no deterministic GPU version; torch warns instead of stopping (`warn_only`) | nothing to do for a run; GPU results may then differ slightly between repeats, which `python -m pytest tests -q -m gpu` (in `system/`) shows |

To keep pip from upgrading NumPy or pandas in this environment later, run once:

```bash
printf 'numpy<2\npandas<3\n' > "$CONDA_PREFIX/pip-constraints.txt"
conda env config vars set PIP_CONSTRAINT="$CONDA_PREFIX/pip-constraints.txt"
conda deactivate && conda activate ficl
```
