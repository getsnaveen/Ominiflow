import sys
import os
import streamlit.web.cli as stcli
import subprocess


class EasyflowRunner:
    def __init__(self):
        pass

    def streamlit_run(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(this_dir)
        sys.path.append(this_dir)
        sys.argv = ["streamlit", "run", "main.py", "--global.developmentMode=false", "--browser.gatherUsageStats=false"]
        sys.exit(stcli.main())

    
