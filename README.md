# ProjectWork
This repo contains a project work aimed to develop a big data pipeline used to classify faulty Scania motors based on infos provided by two open datasets.
Data previously contained in csv files are stored and then fetched from a kafka topic, preprocessed as dataframes by spark framework and then fed to a pytorch model using
neural network for the sake of classification. ML algorithm runs in parallel on top of the GPU exploting pytorch APIs and CUDA hw architecture, the code written in python will be provided via a notebook file. 

# Guidelines

In order to execute the code, it is required the deployment of the following software:
<ol>
  <li>apache kafka_2.12-0.10.2.0</li>
  <li>spark-3.0.1-bin-hadoop2.7</li>
  <li>hadoop-3.0.1</li>
  <li>jupyter notebook</li>
  <li>Python 3.8</li>
  <li>Nvidia cuda</li>
  <li>Nvidia cuDNN</li>
  <li>Pytorch 1.9.0</li>
</ol>
