These scripts are just used for parsing the output log files of Megatron-Deepspeed:
- `grep_iterations.sh`: will parse the lines containing "iteration" in the log file. These lines contain the informations we need.
- `parse_logs.py`: parses the outputs of `grep_iterations.sh` in order to get all the relevant informations for each iteration: loss, sample_per_second, etc. and also get the relevant information about parellelism from the filename.
