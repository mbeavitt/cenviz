Program to visualise centromere repeats as dotplots, with meaningful colours.

Note: requires this dependency to be installed via pip: https://github.com/mbeavitt/trash-compactor

## Usage

Requires a repeat table CSV in TRASH output format.

```bash
# All chromosomes
python visualize_centromere_2d.py repeats.csv --name "Sample"

# Single chromosome
python visualize_centromere_2d.py repeats.csv --chromosome Chr6 --name "Sample"

# Specific region
python visualize_centromere_2d.py repeats.csv --chromosome Chr6 --start 1000000 --end 8000000
```

Run `python visualize_centromere_2d.py --help` for all options.
