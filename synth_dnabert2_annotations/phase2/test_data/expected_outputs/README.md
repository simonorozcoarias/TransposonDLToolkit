# Expected Outputs

This directory will contain expected outputs from test runs for validation purposes.

## Purpose

When running `scripts/run_test.sh`, the prediction results can be compared against expected outputs stored here to verify:
- Prediction pipeline integrity
- Model inference correctness
- Format consistency

## Contents

After running tests, you may save reference outputs here for future validation:
- GFF3 prediction files
- Evaluation metrics JSON
- Summary statistics

## Usage

```bash
# Run test and save outputs as reference
cd ..
./scripts/run_test.sh

# Optionally copy outputs here as expected reference
cp output/*.gff3 expected_outputs/
cp output/*.json expected_outputs/
```

## Note

This directory is optional and not required for the pipeline to run.
