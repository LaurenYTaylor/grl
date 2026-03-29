nohup sh -c 'parallel -j 8 --results logs/ --joblog job_history_jsrl.log < run_commands_jsrl.txt ;' & 
