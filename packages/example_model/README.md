# Example model

This python package provides a simple, standalone executable model.

To run the model within the `uv` environment:

```bash
uv sync --all-packages
uv run python -m example_model.example_model packages/example_model/tests/model_input.json
```

Which will write `model_output.json` with the results of the run.  You
can specify a different location with the `-o` command line argument.
