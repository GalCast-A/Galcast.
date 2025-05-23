# Gunicorn configuration for Cloud Run
bind = "0.0.0.0:8080"  # Will be overridden by --bind in CMD, but included for clarity
workers = 2  # Suitable for Cloud Run's default 1 vCPU
timeout = 600  # Matches Cloud Run's max request timeout (60 minutes)
loglevel = "debug"  # Matches --log-level debug