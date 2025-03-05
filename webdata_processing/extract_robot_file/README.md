# Processing robots.txt

For RedPajama:
```
input_file = path/to/redpajama_all_urls.csv
output_path = output/for/redpajama
```

For FineWeb-Edu:
```
input_file = path/to/fiwebedu_urls.csv
output_path = output/for/fiwebedu
```

## Extract robots.txt

Run the command:
```
mkdir -p $output_path

python robotparser.py --output_path $output_path --input_file $input_file --input_field url --num_workers 50
```

## Post-process the output
```
python postprocess.py --output_path $output_path 
```

## Analyse errors
```
python analyse_errors.py --log_file $output_path/logs.csv --output $output_path/out_analyse_errors 
```