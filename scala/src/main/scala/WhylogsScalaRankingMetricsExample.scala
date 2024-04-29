// Tested on Databricks cluster running as scala notebook:
// * cluster version: 13.3 (includes Apache Spark 3.4.1, Scala 2.12)
// * installed whylogs jar: https://oss.sonatype.org/service/local/repositories/snapshots/content/ai/whylabs/whylogs-spark-bundle_3.1.1-scala_2.12/0.2.0-b2-SNAPSHOT/whylogs-spark-bundle_3.1.1-scala_2.12-0.2.0-b2-20240429.142812-1-all.jar


import java.time.LocalDateTime
import java.time.ZonedDateTime

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SaveMode, SparkSession}
import com.whylogs.spark.WhyLogs._

// COMMAND ----------

// For demo purposes we will create a time column with yesterday's date, so that Whylabs ingestion sees this as a recent dataset profile
// and it shows up in default dashboard of last 7 days on Whylabs.

def unixTimestampForNumberOfDaysAgo(numDaysAgo: Int): ZonedDateTime = {
    import java.time._
    val numDaysAgoDateTime: LocalDateTime = LocalDateTime.now().minusDays(numDaysAgo)
    val zdt: ZonedDateTime = numDaysAgoDateTime.atZone(ZoneId.of("America/Los_Angeles"))
    zdt
}

val timestamp_two_days_ago = unixTimestampForNumberOfDaysAgo(2)



// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DataTypes
val spark = SparkSession
  .builder()
  .master("local[*, 3]")
  .appName("SparkTesting-" + LocalDateTime.now().toString)
  .config("spark.ui.enabled", "false")
  .getOrCreate()

// Using some examples from the Apache support for RankingMetrics: 
// See https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems
val predictionAndLabelsRDD = spark.sparkContext.parallelize(
      Seq(
        (Array(1, 6, 2, 7, 8, 3, 9, 10, 4, 5), Array(1, 2, 3, 4, 5)),
        (Array(4, 1, 5, 6, 2, 7, 3, 8, 9, 10), Array(1, 2, 3)),
        (Array(1, 2, 3, 4, 5), Array(0, 0, 0, 0, 0))),
      2)

// Now we have an example DataFrame with columns for the predictions and targets
val df = predictionAndLabelsRDD.toDF("predictions", "targets")
df.printSchema()

// Next we create a profiling session to compute RankingMetrics
// This must be a stand alone profiling session that does not compute
// other whylogs metrics. The default of k is 10 if not specified
val session = df.newProfilingSession("RankingMetricsTest") // start a new WhyLogs profiling job
  .withRankingMetrics(predictionField="predictions", targetField="targets", k=Some(2))

// COMMAND ----------

// Replace the following parameters below with your values after signing up for an account at https://whylabs.ai/
// You can find Organization Id on https://hub.whylabsapp.com/settings/access-tokens and the value looks something like: org-123abc
// also the settings page allows you t create new apiKeys which you will need an apiKey to upload to your account in Whylabs
// The modelId below specifies which model this profile is for, by default an initial model-1 is created but you will update this
// if you create a new model here https://hub.whylabsapp.com/settings/model-management
// Note: if you don't specify the timestamp current local time is used as a default.
session.logRankingMetrics(timestamp=timestamp_two_days_ago.toInstant,
          orgId = "replace-with-org-id",
          modelId = "replace-with-model-id",
          apiKey = "replace-with-api-key")
