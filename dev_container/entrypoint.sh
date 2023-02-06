#!/bin/sh
# * The #!/bin/sh shebang specifies that the script should be run with the /bin/sh shell,
#   which is a standard shell in many Unix-like operating systems.

# * The file is executed from the WORKDIR directory as defined in the Dockerfile so the path to files are relative
#   to WORKDIR and not to the file's position

# * Notice that the EOL character of this file must be UNIX style (LF) otherwise you will get an error when the
#   script runs in the container

# I use the cache_dir argument instead of setting the HUGGINGFACE_HUB_CACHE environment variable (couldn't make it work)
# set the huggingface cache directory from which the models are downloaded and read from. We set it to the path
# in which we stored the models in the docker image so that we don't have to download them again in the container.
#export HUGGINGFACE_HUB_CACHE='/usr/src/model_scheduler_src/ml_models/huggingface/diffusers/'

# set -e sets a shell option to immediately exit if any command being run exits with a non-zero exit code.
# The script will return with the exit code of the failing command.
set -e

# the main command to run when the container starts.
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# It basically takes all the extra command line arguments and execs them as a command. The intention is basically
# "Do everything in this .sh script, then in the same shell run the command the user passes in on the command line".
exec "$@"