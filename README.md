# Speech-to-Text Application

Deployment URL (Task 7):
``` text
http://speech-Appli-zUWoLIz6kwpm-62646098.ap-southeast-1.elb.amazonaws.com:8080
```

## Prerequisites

Before starting, ensure you have the following installed:
- Docker and Docker Compose
- Python

## Task 2

### Data Files Location

The required data files are located in the `data` directory:

* `cv-valid-dev.csv` - Original data file
* `cv-valid-dev` folder - Contains the full set of MP3 files
* `cv-valid-dev-sample` folder - Contains a small subset of MP3 files for quick testing

Please maintain the original file structure for proper script execution.

### Prerequisites

Before running the Python scripts, set up your environment:

``` bash
cd asr
python3 -m venv venv   
source venv/bin/activate
pip install -r requirements.txt
```

### Execution Order

1. Start the ASR API first
2. Process the CSV file using the CV decode script

### Running the ASR API

Start the inference API:

``` bash
python asr_api.py
```

### Part b: Testing the API

#### Test Ping Endpoint

``` bash
curl --location 'http://localhost:8001/ping'
```

Expected response:
```json
"pong"
```

#### Part c: Test Inference Endpoint

``` bash
curl -F 'file=@"/home/fred/speech-to-text-app/asr/data/cv-valid-dev/sample-000000.mp3"' http://localhost:8001/asr
```

Example response:

``` json
{
    "transcription": "be careful with your prognostications said the stranger",
    "duration": 5.1
}
```

### Part d: Running CV Decode Script

**Important**: Ensure the ASR API is running before executing this script.

**Note**: 
- The default directory where the MP3 files will be procesessed is set as `data/cv-valid-dev-sample`. 
- After the `cv-valid-dev.csv` input file has been successfully processed, it will automatically be deleted from the `data` directory. 
- A new `cv-valid-dev.csv` file will be created in the `asr` directory containing the transcribed text in the generated_text column.

Execute this script to transcribe the MP3 files and populate the results into `cv-valid-dev.csv`.

``` bash
python cv-decode.py
```

### Part e: Docker Setup

- Choose the appropriate commands based on your machine architecture. This will build and run a Docker container using the Dockerfile found in the `asr` directory.

#### For Mac M1 and Above (ARM64)

``` bash
docker build --no-cache --platform linux/arm64 -t asr-api .
docker run -d -p 8001:8001 --name asr-api asr-api
```

#### For AMD Machines with NVIDIA GPU Support

``` bash
docker build --no-cache --platform linux/amd64 -t asr-api .
docker run -d --gpus all -p 8001:8001 --name asr-api asr-api
```

#### Docker Cleanup Commands

``` bash
docker ps
docker stop <container_id_or_name>
docker rm <container_id_or_name>
docker images
docker rmi <image_id_or_name>
```

## Task 3

For task 3, the chosen public cloud to deploy the service is AWS. The proposed architecture for elasticsearch backend and search-ui frontend web application can be found in the "deployment-design" directory as "design.pdf".


## Task 4

**Note**: 
- The data related file "cv-valid-dev.csv" can be found in the "data" directory. 
- The "cv-valid-dev.csv" file is the final processed CSV file from task 2, which has the duration and generated_text fields populated. This file should remain in its original location for the scripts to work correctly.

### Prerequisites

``` bash
cd elastic-backend
python3 -m venv backend_env
source backend_env/bin/activate
pip install -r requirements.txt
```

### Starting Elasticsearch Cluster

Execute these commands to start the 2-node elasticsearch cluster and index cv-valid-dev.csv into the elasticsearch service.

``` bash
docker-compose up -d
python cv-index.py
```

### Verifying Cluster Status

Check that the cluster is running properly with these commands:

``` bash
curl http://localhost:9200/_cluster/health
curl http://localhost:9200/cv-transcriptions/_count
```

Example response for cluster health:

```json
{
    "cluster_name": "es-cluster",
    "status": "green",
    "timed_out": false,
    "number_of_nodes": 2,
    "number_of_data_nodes": 2,
    "active_primary_shards": 2,
    "active_shards": 4,
    "relocating_shards": 0,
    "initializing_shards": 0,
    "unassigned_shards": 0,
    "delayed_unassigned_shards": 0,
    "number_of_pending_tasks": 0,
    "number_of_in_flight_fetch": 0,
    "task_max_waiting_in_queue_millis": 0,
    "active_shards_percent_as_number": 100.0
}
```

Example response for document count:

```json
{
    "count": 4076,
    "_shards": {
        "total": 2,
        "successful": 2,
        "skipped": 0,
        "failed": 0
    }
}
```

### Querying Sample Data

Retrieve a sample of the indexed data:

``` bash
curl -X GET "localhost:9200/cv-transcriptions/_search?size=1&pretty"
```
Example response:
``` json
{
  "took" : 96,
  "timed_out" : false,
  "_shards" : {
    "total" : 2,
    "successful" : 2,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : {
      "value" : 4076,
      "relation" : "eq"
    },
    "max_score" : 1.0,
    "hits" : [
      {
        "_index" : "cv-transcriptions",
        "_id" : "1",
        "_score" : 1.0,
        "_source" : {
          "filename" : "cv-valid-dev/sample-000001.mp3",
          "text" : "then why should they be surprised when they see one",
          "up_votes" : 2,
          "down_votes" : 0,
          "generated_text" : "then why should they be surprised when they see one"
        }
      }
    ]
  }
}
```

### Cleanup

Stop the containers and deactivate the virtual environment:

``` bash
docker-compose down -v
deactivate
```

## Task 5

Before proceeding with the frontend setup, ensure the Elasticsearch backend cluster from Task 4 is running in the background. This is important for the application to function properly.

#### Building and Running the Container

Run these commands to build and start the frontend web application for search-ui:

``` bash
cd search-ui
docker compose build && docker compose up -d
```

#### Accessing the Application

Once the web application is running, you can access it through your browser at:

``` text
http://localhost:3000/
```

#### Stopping the Container

To stop the container and remove associated volumes, use:

``` bash
docker-compose down -v
```

## Task 6 & 7

The search-ui frontend web application has been successfully deployed on AWS and can be accessed through the following URL:

``` text
http://speech-Appli-zUWoLIz6kwpm-62646098.ap-southeast-1.elb.amazonaws.com:8080
```

This deployment is publicly accessible.

## Task 8

My essay submission can be found in the directory under the file `essay.pdf`. The words count is less than 500, excluding headers and figure description.

