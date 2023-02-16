This service is used only in the local development environment to test the text-to-image inference code. 
It is a fast api service. To avoid downloading the stable diffusion model during start up, the model should be 
included in the repository and the container. But because this repository is only used for local debugging, 
it doesn't include the model, to reduce the size of the repo. In the local development environment, 
a docker volume is used to mount the folder with the model in the container. The folder is called 'ml_models' and 
lies directly under the root folder of the repository (the 'model_scheduler_src' folder). 

