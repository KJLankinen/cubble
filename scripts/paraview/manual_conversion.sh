# This is a script that shows how to manually convert snapshot files from CSV to VTU
# Assume the CSV files are located in folders called run_0, run_1 etc
# Inside /scratch/work/ulanove1/cubble/data/16_08_2019/sample_run
# First activates virtual environment (needs to be installed first, see instructions when running array_job.py
# for first time)
# Passes location to conversion script
# Deactivates virtual environment
source scripts/cubble_python_venv/bin/activate
python scripts/convert_csv_to_vtu.py --snapshot_dir=/scratch/work/ulanove1/cubble/data/16_08_2019/sample_run -si
deactivate
