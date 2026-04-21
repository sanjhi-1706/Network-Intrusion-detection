from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# -------------------------------
# 1. Create Output Folder
# -------------------------------
os.makedirs("eda_output", exist_ok=True)

# -------------------------------
# 2. Start Spark Session
# -------------------------------
spark = SparkSession.builder \
    .appName("NIDS_EDA") \
    .master("local[*]") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
# -------------------------------
# 3. Load Data
# -------------------------------
data = spark.read.csv(
    "hdfs://master:9000/nids/part_*",
    header=False
)

# Reduce heavy load early
data = data.sample(fraction=0.05)
data = data.repartition(10)

print("✅ Data Loaded")

# -------------------------------
# 4. Rename Columns
# -------------------------------
num_cols = len(data.columns)
new_cols = [f"col_{i}" for i in range(num_cols - 1)] + ["Label"]
data = data.toDF(*new_cols)

# -------------------------------
# 5. Convert Data Types
# -------------------------------
for c in data.columns[:-1]:
    data = data.withColumn(c, col(c).cast("double"))

# Remove null rows after casting
data = data.dropna()

print("✅ Data cleaned & converted")

# -------------------------------
# 6. Save Basic Info
# -------------------------------
with open("eda_output/eda_summary.txt", "w") as f:
    f.write("===== BASIC INFO =====\n")
    f.write(f"Total Columns: {len(data.columns)}\n")
    f.write(f"Column Names: {data.columns}\n\n")

    f.write("===== SAMPLE DATA =====\n")
    for row in data.limit(5).collect():
        f.write(str(row) + "\n")

print("✅ Basic info saved")

# -------------------------------
# 7. Class Distribution
# -------------------------------
class_df = data.groupBy("Label").count()
class_df.show()

class_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
    "eda_output/class_distribution"
)

print("✅ Class distribution saved")

# -------------------------------
# 8. Summary Statistics
# -------------------------------
summary_df = data.select("col_0", "col_1", "col_2").describe()
summary_df.show()

summary_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
    "eda_output/summary_stats"
)

print("✅ Summary stats saved")

# -------------------------------
# 9. Safe Sampling for Pandas
# -------------------------------
sample_df = data.sample(fraction=0.1).limit(500).toPandas()

print("✅ Sample created")

# -------------------------------
# 10. Save Sample CSV
# -------------------------------
sample_df.to_csv("eda_output/sample_data.csv", index=False)

print("✅ Sample data saved")

# -------------------------------
# 11. Class Distribution Plot
# -------------------------------
plt.figure(figsize=(10, 6))
sns.countplot(x="Label", data=sample_df)
plt.xticks(rotation=90)
plt.title("Attack Distribution")
plt.tight_layout()
plt.savefig("eda_output/class_distribution.png")
plt.close()

# -------------------------------
# 12. Histogram Plot
# -------------------------------
numeric_cols = sample_df.select_dtypes(include=['number']).columns[:5]

if len(numeric_cols) > 0:
    sample_df[numeric_cols].hist(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig("eda_output/feature_distribution.png")
    plt.close()

print("✅ Histogram saved")

# -------------------------------
# 13. Correlation Heatmap
# -------------------------------
corr_cols = sample_df.select_dtypes(include=['number']).columns[:10]

if len(corr_cols) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(sample_df[corr_cols].corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("eda_output/correlation_heatmap.png")
    plt.close()

print("✅ Heatmap saved")

# -------------------------------
# 14. Missing Values Report
# -------------------------------
missing_df = pd.DataFrame({
    'Column': sample_df.columns,
    'Missing Values': sample_df.isnull().sum().values
})

missing_df.to_csv("eda_output/missing_values.csv", index=False)

print("✅ Missing values report saved")

# -------------------------------
# 15. Stop Spark
# -------------------------------
spark.stop()

print("🎯 EDA COMPLETED SUCCESSFULLY")
