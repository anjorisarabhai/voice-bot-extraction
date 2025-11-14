# config/__init__.py
from dotenv import load_dotenv

# CRITICAL: Load environment variables immediately upon importing the 'config' package.
load_dotenv()

# Import all settings from the settings file
from .settings import *