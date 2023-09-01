# Private Record Linkage

This repository contains software for computing record linkage among multiple clients while preserving user data privacy.

* Use pre-trained large language models with contrastive fine-tuning to encode client records into fixed-length embeddings.
* Clients encrypt and send the embeddings to the server. 
* Server applies Berlekamp-Welch algorithm to efficiently compute the matching records without learning additional information.
* Client encoding and server matching using Berlekamp-Welch can be done using 16 or 32 bit field arithmetic. Switching is done using a flag. 
* Server side field arithmetic uses lookup tables, which are auto-generated when running the server code (a one-time process). Tables can be generated beforehand to skip the auto-generation delay using a generation command.
* Server returns the results back to clients.

## Requirement

* Python=3.9
* Go=1.19
* System requirements:
  * Server:
    * RAM: The server requires 64 GB memory for the 32-bit version.
    * CPU: To take advantage of parallel computation, machines with multicore processors are encouraged to be used.
  * Client:
    * GPU: GPUs can be used to speed up the encoding process, but are not required.


## Install and compile

* Clone the repo:
```
git clone https://github.com/AON-PRISMA/feaas.git
cd feaas
```

### Client

* Create a Python environment and install Python 3.9

* Install required libraries:
```
pip install -r requirements.txt
```

* Build the Go component on the client-side:
```
cd client/share_gen/
# for 32-bit version
go build -buildmode=c-shared -o _share_gen.so

# for 16-bit version
go build -tags 16 -buildmode=c-shared -o _share_gen.so
```

### Server

* Install required libraries:
```
cd ../../server/
go get
```

* Build look-up tables:
```
# for 32-bit version
go run tablegen.go -num_bits=32

# for 16-bit version
go run tablegen.go -num_bits=16
```


* Build server code:
```
# for 32-bit version
go build -o server server.go

# for 16-bit version
go build -tags 16 -o server server.go
```

* To build the client and server using one command (first cd into feaas/server/):
```
sh build.sh [16|32]
```


## Usage


### Server
```shell
cd feaas/server/
./server -num_c=[NUM_C] -rep=[REP] -lsh_rep=[LSH_REP] -tls=[TLS] -addr=[ADDR] -output=[OUTPUT]

arguments:
    NUM_C:      total number of participating clients
    REP:        number of repetitions for the encryption to improve security
    LSH_REP:    repetition param for lsh; negative num means no use of lsh
    TLS:        whether to enable tls; 1 or 0
    ADDR:       ip address the server is listening (default value: 127.0.0.1:8000) 
    OUTPUT:     the output csv path for storing matching results
```


### Client

#### 1. Determine the threshold and generate data encodings

```shell
cd feaas/encoding
python encode.py --load_config [CONFIG_FILE] -d [DATA_PATH] --data_name [DATA_NAME] -r [RATIO] -n [MODEL_NAME] -p [CKPT_PATH] -i [NUM_INTERVAL] --bs [BS] -s [SAVE_PATH]

arguments:
    CONFIG_FILE:   the config file containing default command-line arguments; arguments are overridden if provided explicitly
    DATA_PATH:     path of the data file; expect a .csv file
    DATA_NAME:     dataset name; one of ['ag', 'febrl4', 'abt_buy']  
    RATIO:         ratio (num of negatives /num of positives) of the dataset for selecting the threshold
    MODEL_NAME:    model architecture name for inference
    CKPT_PATH:     path of the model checkpoint 
    NUM_INTERVAL:  number of intervals for quantization (converting embeddings to integer vectors)
    BS:            batch size used during inference
    SAVE_PATH:     path of the encodings output

output:
    Threshold for decision-making in later steps
```

#### 2. Encrypt and send
```shell
cd feaas/client/
python main.py --load_config [CONFIG_FILE] --cid [CID] --server_addr [SERVER_ADDR] --client_addr [CLIENT_ADDR] --tls_file [TLS_FILE] --data_path [DATA_PATH]  --th [TH] --rep [REP] --lsh_rep [LSH_REP] --lsh_num_ind [LSH_NUM_IND] --lsh_num_bin [LSH_NUM_BIN]
--num_bits [NUM_BITS]

arguments:
    CONFIG_FILE:   the config file containing default command-line arguments; arguments are overridden if provided explicitly
    CID:           current client id, has to be non-repetitive and starts from 1
    SERVER_ADDR:   ip address to connect to the server (default value: 127.0.0.1:8000)
    CLIENT_ADDR:   ip address used between clients (default value: 127.0.0.1:7000)
    TLS_FILE:      path to the tls file for verification; if not provided, non-tls is used
    DATA_PATH:     dataset path, expect a .npy file
    TH:            threshold for decision-making (output from Step 1)
    REP:           number of repetitions for encryption to improve security
    LSH_REP:       repetition param for lsh; negative num means no use of lsh
    LSH_NUM_IND:   number of selected indices for lsh
    LSH_NUM_BIN:   number of bins for lsh
    NUM_BITS:      number of bits of the protocol; the clients and server should use the same NUM_BITS version
```

  * **Note**: Client 1 with (cid=1) serves as the coordinator among clients and needs to be started before any other clients.

## Examples

The commands below perform an end-to-end run of the 32-bit version on the Amazon-Google dataset. For demonstration purposes, two clients communicate with a server locally. The matching output is stored in server/result.csv.

* Compile the code:
```shell
cd feaas/server/
sh build.sh 32
```

### Server
* Start the server and wait for clients' connection. Note: the initialization of the server might take a few minutes to complete for the **32-bit** version.
```shell
./server -num_c=2 -rep=1 -lsh_rep=10 -tls=0 -output=result.csv
```


### Client 1

* Determine the threshold from a given dataset and generate data embeddings from querying the pre-trained model:
```shell
cd feaas/encoding/
python encode.py --load_config ../data/amazon_google/encoding_config.json -d ../data/amazon_google/test_df_1.csv -s ../client/embed_1.npy
```
* Start the client application:
  * **Note:** Clients may need to explicitly set the num_bits arguments or modify the config.json file to match the server's version.
```shell
cd ../client/
python main.py --cid 1 --load_config ../data/amazon_google/config.json --data_path embed_1.npy
```

### Client 2
* Open up a new terminal. This is similar to client 1, but we use client 2's data records.
```shell
cd feaas/encoding/
python encode.py --load_config ../data/amazon_google/encoding_config.json -d ../data/amazon_google/test_df_2.csv -s ../client/embed_2.npy
cd ../client/
python main.py --cid 2 --load_config ../data/amazon_google/config.json --data_path embed_2.npy
```


### Datasets
The example dataset is downloaded and adapted from:
https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution

## License
See the [LICENSE](LICENSE.txt) file for license rights and limitations.
