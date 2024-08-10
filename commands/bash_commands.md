# Essential Bash Commands for Data Scientists

1. `ls`
    ```bash    
   # List directory contents. Useful for navigating data directories.
   ls -l data/
    ```

2. `cd`
    ```bash
   # Change directory. Essential for navigating file systems.
   cd /path/to/data/folder
    ```

3. `pwd`
    ```bash
   # Print working directory. Helpful to confirm current location.
   pwd
    ```

4. `mkdir`
    ```bash
   # Make directory. Create new folders for organizing data or projects.
   mkdir new_project
    ```

5. `rm`
    ```bash
   # Remove files or directories. Use with caution!
   rm olddata.csv
    ```

6. `cp`
    ```bash
   # Copy files or directories.
   cp data.csv backup_data.csv
    ```

7. `mv`
    ```bash
   # Move or rename files or directories.
   mv old_name.txt new_name.txt
    ```

8. `cat`
    ```bash
   # Concatenate and display file content. Quick way to view data files.
   cat sample_data.csv
    ```

9. `head`
    ```bash
   # Display the beginning of a file. Useful for previewing data files.
   head -n 10 large_dataset.csv
    ```

10. `tail`
    ```bash
    # Display the end of a file. Good for checking recent log entries.
    tail -n 20 experiment_log.txt
    ```

11. `less`
    ```bash
    # View file contents page by page. Great for large files.
    less huge_dataset.csv
    ```

12. `grep`
    ```bash
    # Search for patterns in files. Essential for data exploration.
    grep "error" log_file.txt
    ```

13. `awk`
    ```bash
    # Pattern scanning and text processing. Powerful for data manipulation.
    awk '{print $1,$4}' data.txt
    ```

14. `sed`
    ```bash
    # Stream editor for filtering and transforming text.
    sed 's/old/new/g' file.txt
    ```

15. `wc`
    ```bash
    # Word, line, character, and byte count. Useful for quick data stats.
    wc -l dataset.csv
    ```

16. `sort`
    ```bash
    # Sort lines of text. Helpful for ordering data.
    sort -n numeric_data.txt
    ```

17. `uniq`
    ```bash
    # Report or filter out repeated lines in a file. Good for finding unique values.
    sort data.txt | uniq -c
    ```

18. `cut`
    ```bash
    # Remove sections from each line of files. Extract specific columns.
    cut -d',' -f1,3 data.csv
    ```

19. `tr`
    ```bash
    # Translate or delete characters. Useful for data cleaning.
    cat file.txt | tr '[:lower:]' '[:upper:]'
    ```

20. `find`
    ```bash
    # Search for files in a directory hierarchy. Locate specific data files.
    find . -name "*.csv"
    ```

21. `xargs`
    ```bash
    # Build and execute command lines from standard input. Powerful for batch operations.
    find . -name "*.txt" | xargs grep "pattern"
    ```

22. `tar`
    ```bash
    # Tape archiver. Compress or extract files, often used for datasets.
    tar -czvf archive.tar.gz data_folder/
    ```

23. `gzip` / `gunzip`
    ```bash
    # Compress or decompress files. Common for large datasets.
    gzip large_file.csv
        ```gunzip 
    large_file.csv.gz

24. `curl`
    ```bash
    # Transfer data from or to a server. Download datasets from URLs.
    curl -O https://example.com/dataset.csv
    ```

25. `wget`
    ```bash
    # Non-interactive network downloader. Alternative to curl for downloading data.
    wget https://example.com/large_dataset.zip
    ```

26. `ssh`
    ```bash
    # Secure shell. Connect to remote servers or clusters.
    ssh username@remote_server
    ```

27. `scp`
    ```bash
    # Secure copy. Transfer files between local and remote machines.
    scp local_file.csv username@remote_server:/path/to/destination/
    ```

28. `screen` / `tmux`
    ```bash
    # Terminal multiplexers. Run long processes in the background.
    screen
        ```tmux


29. `chmod`
    ```bash
    # Change file mode bits. Modify permissions, often needed for scripts.
    chmod +x run_analysis.sh
    ```

30. `echo`
    ```bash
    # Display a line of text. Useful in scripts and for debugging.
    echo "Processing complete"
    ```

31. `pipe (|)` and `redirect (>, >>)`
    ```bash
    # Combine commands and redirect output. Essential for data processing workflows.
    cat data.csv | grep "pattern" > filtered_data.csv
        ```echo "N
    ew entry" >> log.txt