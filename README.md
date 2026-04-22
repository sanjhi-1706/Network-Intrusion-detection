# 🛡️ Network Intrusion Detection System (NIDS) with Big Data

A deep learning-based Network Intrusion Detection System built on a distributed Hadoop + Spark cluster, trained on the CICIDS-2017 dataset using TensorFlow/CNN.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [System Architecture](#system-architecture)
4. [Prerequisites](#prerequisites)
5. [Hadoop Cluster Setup](#hadoop-cluster-setup)
6. [Hadoop Configuration](#hadoop-configuration)
7. [Spark Cluster Setup](#spark-cluster-setup)
8. [Dataset Upload to HDFS](#dataset-upload-to-hdfs)
9. [Jupyter Notebook Integration](#jupyter-notebook-integration)
10. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
11. [CSV to Parquet Conversion](#csv-to-parquet-conversion)
12. [Model Training](#model-training)
13. [Project Flow](#project-flow)
14. [Directory Structure](#directory-structure)
15. [Troubleshooting](#troubleshooting)

---

## 📌 Project Overview

This project implements a **Network Intrusion Detection System (NIDS)** using a Big Data pipeline. The system:

- Ingests large-scale network traffic data (~2.8 million records)
- Stores and distributes data across a multi-node **Hadoop HDFS** cluster
- Processes data in parallel using **Apache Spark**
- Trains a **Convolutional Neural Network (CNN)** via **TensorFlow** to classify network traffic as benign or malicious
- Covers multiple attack types including DoS, Slowloris, and web-based attacks

---

## 📂 Dataset

**Source:** [CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) — specifically the **Thursday Morning Web Attacks** subset.

| Property | Details |
|---|---|
| Records | ~2.8 million |
| Features | ~79 columns |
| Target Column | `Label` (Benign / Attack type) |
| Format | CSV |

### Attack Classes

- DoS Hulk
- DoS GoldenEye
- Slowloris
- SlowHTTPTest
- Web Attacks (XSS, SQL Injection, Brute Force)

### Key Features Used

| Feature | Description |
|---|---|
| Source IP | Source IP address |
| Destination IP | Destination IP address |
| Flow Duration | Duration of the network flow |
| Total Fwd Packets | Packets sent in forward direction |
| Total Backward Packets | Packets sent in backward direction |
| Packet Length Mean | Average packet length |
| Flow Bytes/s | Flow rate in bytes per second |
| Flow Packets/s | Flow rate in packets per second |
| SYN Flag Count | Number of SYN flags |
| ACK Flag Count | Number of ACK flags |
| Label | Target class (Benign or Attack) |

---

## 🏗️ System Architecture

```
Jupyter Notebook (Driver)
        │
        ▼
  Spark Driver
        │
        ▼
Spark Master (port 7077)
        │
   ┌────┴────┐
   ▼         ▼
Spark     Spark
Worker    Worker
(slave1)  (slave2)
   │         │
   ▼         ▼
HDFS      HDFS
DataNode  DataNode
   │         │
   └────┬────┘
        ▼
  Dataset Blocks
  (Replicated, Distributed)
```

### Node Roles

| Node | Role | Services |
|---|---|---|
| `master` (10.200.218.49) | Master | NameNode, ResourceManager, SecondaryNameNode, Spark Master |
| `slave1` (10.200.218.50) | Worker | DataNode, NodeManager, Spark Worker |
| `slave2` (10.200.218.51) | Worker | DataNode, NodeManager, Spark Worker |

---

## ✅ Prerequisites

- Ubuntu Linux on all nodes
- Java 8 or 11
- Hadoop 3.x
- Apache Spark 3.3.2 (built for Hadoop 3)
- Python 3.x
- PySpark
- TensorFlow 2.x
- Jupyter Notebook
- SSH access between all nodes

---

## 🖥️ Hadoop Cluster Setup

### Step 1: Set Hostnames

On each respective machine, set the hostname:

```bash
# On master node
sudo hostnamectl set-hostname master

# On first worker
sudo hostnamectl set-hostname slave1

# On second worker
sudo hostnamectl set-hostname slave2
```

### Step 2: Update `/etc/hosts` on ALL Machines

Add the following entries to `/etc/hosts` on **master, slave1, and slave2**:

```
10.200.218.49 master
10.200.218.50 slave1
10.200.218.51 slave2
```

This allows nodes to communicate using hostnames instead of IP addresses.

### Step 3: Configure Passwordless SSH (from Master)

```bash
# Generate SSH key pair on master
ssh-keygen -t rsa

# Copy public key to slave nodes
ssh-copy-id slave1
ssh-copy-id slave2

# Verify connection
ssh slave1
ssh slave2
```

> ⚠️ Passwordless SSH is required for Hadoop and Spark to automatically start/stop services on worker nodes.

---

## ⚙️ Hadoop Configuration

All configuration files are located in `$HADOOP_HOME/etc/hadoop/`.

### `core-site.xml`

Defines the HDFS master URI:

```xml
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://master:9000</value>
  </property>
</configuration>
```

### `hdfs-site.xml`

Defines replication factor and storage paths:

```xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>2</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file:///home/bda/hadoopdata/hdfs/namenode</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file:///home/bda/hadoopdata/hdfs/datanode</value>
  </property>
</configuration>
```

### `yarn-site.xml`

Configures YARN resource management:

```xml
<configuration>
  <property>
    <name>yarn.resourcemanager.hostname</name>
    <value>master</value>
  </property>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
</configuration>
```

### `workers` file

```
slave1
slave2
```

---

## 🚀 Starting Hadoop Services

```bash
# Format NameNode (first time only)
hdfs namenode -format

# Start HDFS
start-dfs.sh

# Start YARN
start-yarn.sh
```

### Verify Running Processes with `jps`

**On master:**

```
NameNode
SecondaryNameNode
ResourceManager
Master
Worker
```

**On slave nodes:**

```
DataNode
NodeManager
Worker
```

> ℹ️ Access the Hadoop Web UI at: `http://master:9870` (HDFS) and `http://master:8088` (YARN)

---

## ⚡ Spark Cluster Setup

### Start Spark Master (on master node)

```bash
cd ~/spark3/spark-3.3.2-bin-hadoop3
sbin/start-master.sh
```

### Start Spark Workers (on master node, targets slaves)

```bash
sbin/start-worker.sh spark://master:7077
```

### Verify Spark is Running

```bash
jps
# Expected: Master, Worker

ss -tulnp | grep 7077
# Expected: LISTEN on port 7077
```

**Spark Master URL:** `spark://master:7077`

> ℹ️ Access Spark Web UI at: `http://master:8080`

---

## 📤 Dataset Upload to HDFS

### 1. Create a Directory in HDFS

```bash
hdfs dfs -mkdir /nids
```

### 2. Upload Dataset Files

```bash
hdfs dfs -put part_* /nids/
```

### 3. Verify Upload

```bash
hdfs dfs -ls /nids
```

**HDFS Path:** `hdfs://master:9000/nids/part_*`

### 4. Check Block Distribution

```bash
hdfs fsck /nids -files -blocks -locations
```

### How HDFS Partitions the Data

Files are split into 128 MB blocks and distributed across DataNodes:

| Block | Stored On | Replica |
|---|---|---|
| `part_a` block 1 | slave1 | slave2 |
| `part_a` block 2 | slave2 | slave1 |
| `part_b` block 1 | slave1 | slave2 |
| `part_b` block 2 | slave2 | slave1 |

**Benefits:**
- Parallel processing across nodes
- Fault tolerance via replication (factor = 2)
- Faster data reads

---

## 📓 Jupyter Notebook Integration

The Jupyter Notebook acts as the **Spark Driver**. Configure environment variables before creating a SparkSession:

```python
import os
import sys

os.environ["SPARK_HOME"] = "/home/bda/spark3/spark-3.3.2-bin-hadoop3"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"
os.environ["SPARK_MASTER"] = "spark://10.200.218.49:7077"

sys.path.append("/home/bda/spark3/spark-3.3.2-bin-hadoop3/python")
sys.path.append("/home/bda/spark3/spark-3.3.2-bin-hadoop3/python/lib/py4j-0.10.9.5-src.zip")
```

### Create SparkSession

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("NIDS") \
    .master("spark://10.200.218.49:7077") \
    .config("spark.driver.host", "10.200.218.49") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://master:9000") \
    .enableHiveSupport() \
    .getOrCreate()
```

---

## 🔄 Data Loading and Preprocessing

### Load CSV from HDFS

```python
df = spark.read.csv(
    "hdfs://master:9000/nids/part_*",
    header=False,
    inferSchema=True
)
```

### Rename Columns

```python
num_cols = len(df.columns)
new_cols = [f"col_{i}" for i in range(num_cols - 1)] + ["Label"]
df = df.toDF(*new_cols)
```

> The last column is always renamed to `Label` (the target class).

---

## 🗜️ CSV to Parquet Conversion

Parquet is the preferred storage format for Spark-based workflows.

| Property | CSV | Parquet |
|---|---|---|
| Schema stored | ❌ No | ✅ Yes |
| Storage size | Large | Compact |
| Read speed | Slow | Fast |
| Column pruning | ❌ No | ✅ Yes |
| Spark compatibility | Basic | Optimized |

### Convert and Save as Parquet

```python
df.write.mode("overwrite").parquet("hdfs://master:9000/nids/parquet_data")
```

### Load from Parquet (subsequent runs)

```python
df = spark.read.parquet("hdfs://master:9000/nids/parquet_data")
```

---

## 🤖 Model Training

After preprocessing:

1. **Data cleaning** — handle nulls, drop irrelevant features, encode labels
2. **Sampling** — balance classes for deep learning
3. **Model** — CNN built with TensorFlow/Keras
4. **Training** — distributed via Spark workers
5. **Evaluation** — accuracy, precision, recall, F1-score

---

## 🔁 Project Flow

```
CICIDS-2017 CSV Dataset
         │
         ▼
   Upload to HDFS
         │
         ▼
Distributed Storage on DataNodes
(128 MB blocks, replication factor = 1)
         │
         ▼
  Spark reads HDFS data
         │
         ▼
Data Cleaning + Preprocessing
         │
         ▼
  Convert CSV → Parquet
         │
         ▼
  Sampling for deep learning
         │
         ▼
TensorFlow / MLP model training
         │
         ▼
  Prediction and Evaluation
```

---

## 🐛 Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `ssh: Connection refused` on slave | SSH not configured | Re-run `ssh-copy-id slaveX` |
| NameNode not starting | Already formatted with different cluster ID | Delete `namenode` dir and reformat |
| Spark worker not connecting | Wrong master IP | Check `SPARK_MASTER` env variable |
| HDFS path not found | Files not uploaded | Re-run `hdfs dfs -put` command |
| `py4j` import error | Wrong sys.path | Verify the py4j `.zip` path matches Spark version |
| Parquet write fails | Insufficient HDFS space | Check `hdfs dfsadmin -report` for space |

### Useful Commands

```bash
# Check HDFS health
hdfs dfsadmin -report

# Stop all Hadoop services
stop-dfs.sh && stop-yarn.sh

# Stop Spark
sbin/stop-all.sh

# View HDFS logs
cat $HADOOP_HOME/logs/hadoop-*-namenode-*.log
```

---

## 👥 Team

Big Data Analytics (BDA) Project — Hadoop + Spark + TensorFlow NIDS Pipeline

---

## 📄 License

This project is for academic and research purposes. The CICIDS-2017 dataset is provided by the Canadian Institute for Cybersecurity (CIC).
