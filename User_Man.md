# deepTFBS User Manual (version 1.0)

- **deepTFBS** takes advantages of **multi-task learning** technique to integrate large-scale TF binding profiles for **pre-training**, and is capable of leveraging knowledge from pre-trained models via **transfer learning**, representing an innovation in that it can improve prediction accuracy of **TFBS** under small-sample training and cross-species prediction tasks.
- The deepTFBS project is hosted on https://github.com/cma2015/deepTFBS.
- The deepTFBS Docker image can be obtained from https://hub.docker.com/r/malab/deeptfbs.
- The following part shows installation of deepTFBS docker image and detailed documentation for each function in deepTFBS.

## deepTFBS installation

#### **Step 1**: Docker installation

**i) Docker installation and start ([Official installation tutorial](https://docs.docker.com/install))**

For **Windows (Only available for Windows 10 Prefessional and Enterprise version):**

- Download [Docker](https://download.docker.com/win/stable/Docker for Windows Installer.exe) for windows;
- Double click the EXE file to open it;
- Follow the wizard instruction and complete installation;
- Search docker, select **Docker for Windows** in the search results and click it.

For **Mac OS X (Test on macOS Sierra version 10.12.6 and macOS High Sierra version 10.13.3):**

- Download [Docker](https://download.docker.com/mac/stable/Docker.dmg) for Mac OS;

- Double click the DMG file to open it;
- Drag the docker into Applications and complete installation;
- Start docker from Launchpad by click it.

For **Ubuntu (Test on Ubuntu 18.04 LTS):**

- Go to [Docker](https://download.docker.com/linux/ubuntu/dists/), choose your Ubuntu version, browse to **pool/stable** and choose **amd64, armhf, ppc64el or s390x**. Download the **DEB** file for the Docker version you want to install;
- Install Docker, supposing that the DEB file is download into following path: *"/home/docker-ce~ubuntu_amd64.deb"*

```shell
  $ sudo dpkg -i /home/docker-ce<version-XXX>~ubuntu_amd64.deb      
  $ sudo apt-get install -f
```

**ii) Verify if Docker is installed correctly**

Once Docker installation is completed, we can run `hello-world` image to verify if Docker is installed correctly. Open terminal in Mac OS X and Linux operating system and open CMD for Windows operating system, then type the following command:

```
 $ docker run hello-world
```

**Note:** root permission is required for Linux operating system.

#### **Step 2**: deepTFBS installation from Docker Hub

```shell
# pull latest deepTFBS Docker image from docker hub
$ docker pull malab/deeptfbs
```

#### Step 3: Launch deepTFBS local server

```shell
$ docker run -it malab/deeptfbs bash
$ source activate
$ conda activate deepTFBS
```

Then, deepTFBS framework can be accessed.

## deepTFBS Feature Encoding

This module provides  one-hot encoding strategy (see following table for details).

| Strategy | Description                                                  | Input                     | Output                                                      |
| -------- | ------------------------------------------------------------ | ------------------------- | ----------------------------------------------------------- |
| one-hot  | one-hot encoding was used to transfer nucleotides into numerical arrays: in which an *A* is encoded by [1,0,0,0], a *C* is encoded by [0,1,0,0], a *G* is encoded by [0,0,1,0], a *T* is encoded by [0,0,0,1]. | Sequences in FASTA format | Numerical arrays (can be used to train deep learning model) |

#### How to use the function

```python
python deepTFBS.py data_process -h
usage: deepTFBS.py data_process [-h] [-v] [-i INPUT] [-win WINDOW_LENGTH] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -i INPUT, --input INPUT
                        Path to processed data directory.
  -win WINDOW_LENGTH, --window_length WINDOW_LENGTH
                        Window length
  -o OUTPUT, --output OUTPUT
                        Path to output directory
```

## deepTFBS Model Development

This module provides three funcitons (see following table for details) .

| Functions      | Description                                              | Input                                                | Ouput            |
| -------------- | -------------------------------------------------------- | ---------------------------------------------------- | ---------------- |
| Model pretrain | Construct a pretrained model for TFBS                    | Train, valid and test datasets.                      | PreTrained model |
| Model train    | Select a model to transfer learning                      | Train, valid and test datasets  and pretrained model | Trained model    |
| Model predict  | Select a trained model to predict for the input sequence | Numerical arrays and trained model                   | Predicted score  |

#### Model pretrain

#### How to use the function

```python
python deepTFBS.py pre_train -h
usage: deepTFBS.py pre_train [-h] [-v] -i TFBSDIC -tf TF [-eval EVAL_AFTER_TRAIN] [-lr LR_INIT] [-e EPSILON] [-ep N_EPOCHS] [-bs BATCH_SIZE]
                             -o RESDIC

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -i TFBSDIC, --TFBSDic TFBSDIC
                        Path for processed data directory.
  -tf TF, --TF TF       TF name for processed data.
  -eval EVAL_AFTER_TRAIN, --eval_after_train EVAL_AFTER_TRAIN
                        Eval after train
  -lr LR_INIT, --lr_init LR_INIT
                        Initial learning rate
  -e EPSILON, --epsilon EPSILON
                        Initial learning rate
  -ep N_EPOCHS, --n_epochs N_EPOCHS
                        Initial pretraining epochs
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Initial batch size
  -o RESDIC, --resDic RESDIC
                        Path for checkpoint directory
```

#### Model train

#### How to use the function

```
python deepTFBS.py train -h
usage: deepTFBS.py train [-h] [-v] -i TFBSDIC -pm PRETRAIN_MODEL -tf TF [-eval EVAL_AFTER_TRAIN] [-lr LR_INIT] [-e EPSILON] [-ep N_EPOCHS]
                         [-bs BATCH_SIZE] -o RESDIC

.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -i TFBSDIC, --TFBSDic TFBSDIC
                        Path for processed data directory.
  -pm PRETRAIN_MODEL, --pretrain_model PRETRAIN_MODEL
                        name for pretrain model.
  -tf TF, --TF TF       TF name for processed data.
  -eval EVAL_AFTER_TRAIN, --eval_after_train EVAL_AFTER_TRAIN
                        Eval after train
  -lr LR_INIT, --lr_init LR_INIT
                        Initial learning rate
  -e EPSILON, --epsilon EPSILON
                        Initial learning rate
  -ep N_EPOCHS, --n_epochs N_EPOCHS
                        Initial pretraining epochs
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Initial batch size
  -o RESDIC, --resDic RESDIC
                        Path for checkpoint directory
```

#### Model predict

#### How to use the function

```python
python deepTFBS.py predict -h
usage: deepTFBS.py predict [-h] [-v] -i INPUT_NPZ -tf TF [-lr LR_INIT] [-e EPSILON] [-bs BATCH_SIZE] -d RESDIC -o OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -i INPUT_NPZ, --input_npz INPUT_NPZ
                        File path for predicting.
  -tf TF, --TF TF       TF name for processed data.
  -lr LR_INIT, --lr_init LR_INIT
                        Initial learning rate
  -e EPSILON, --epsilon EPSILON
                        Initial learning rate
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Initial batch size
  -d RESDIC, --resDic RESDIC
                        Path for checkpoint directory
  -o OUTPUT, --output OUTPUT
                        Path for output filename
```

