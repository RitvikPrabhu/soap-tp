import csv
import math
import os
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys
import unittest

import torch
import torch.distributed as dist

from soap_tp import soap_step

ROOT = Path(__file__).resolve().parents[2]

