# Use a Python scientific stack with Jupyter as a parent image
FROM jupyter/scipy-notebook:latest

# Add gym_firecommander
ADD /gym_firecommander /home/jovyan/gym_firecommander

# Install any needed packages specified in requirements.txt
COPY requirements.txt /tmp
RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Run Tensorboard and Jupyter-Lab)
# CMD ["tensorboard", "--logdir", "./work/log"]
# CMD ["jupyter-lab"]
