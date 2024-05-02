// Tested on Databricks cluster running as scala notebook:
// * cluster version: 13.3 (includes Apache Spark 3.4.1, Scala 2.12)
// * installed whylogs jar: https://oss.sonatype.org/service/local/repositories/snapshots/content/ai/whylabs/whylogs-spark-bundle_3.1.1-scala_2.12/0.2.0-b4-SNAPSHOT/whylogs-spark-bundle_3.1.1-scala_2.12-0.2.0-b4-20240502.212838-1-all.jar
/* Maven module
<dependency>
  <groupId>ai.whylabs</groupId>
  <artifactId>whylogs-spark-bundle_3.1.1-scala_2.12</artifactId>
  <version>0.2.0-b4-SNAPSHOT</version>
  <classifier>all</classifier>
</dependency>
*/

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


// create a few days so we get different profiles for a daily model using timeColumn
val t1 = unixEpochTimeForNumberOfDaysAgo(1)
val t2 = unixEpochTimeForNumberOfDaysAgo(2)
val t3 = unixEpochTimeForNumberOfDaysAgo(3)
val timeColumn = "dataset_timestamp"



// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DataTypes
val spark = SparkSession
  .builder()
  .master("local[*, 3]")
  .appName("SparkTesting-" + LocalDateTime.now().toString)
  .config("spark.ui.enabled", "false")
  .getOrCreate()

// Adapting examples from the Apache support for RankingMetrics: 
// See https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems
val predictionAndLabelsAndeScoresWithGroupsRDD = spark.sparkContext.parallelize(
      Seq(
        ("g1", Array(1, 6, 2, 7, 8, 3, 9, 10, 4, 5), Array(1, 2, 3, 4, 5)),
        ("g1", Array(4, 1, 5, 6, 2, 7, 3, 8, 9, 10), Array(1, 2, 3)),
        ("g1", Array(1, 2, 3, 4, 5), null), // test out missing elements for labels
        ("g3", Array(1, 6, 2, 7, 8, 3, 9, 10, 4, 5), Array(1, 2, 3, 4, 5)),
        ("g2", Array(4, 1, 5, 6, 2, 7, 3, 8, 9, 10), Array(1, 2, 3)),
        ("g2", Array(4, 1, 5, 6, 2, 7, 3, 8, 9, 10), Array(1, 2, 3)),
        ("g2", Array(1, 2, 6, 4, 3), null),
        ("g2", Array(1, 2, 3, 4, 5), Array(0, 0, 0, 0, 0)),
        ("g3", Array(1, 6, 2, 7, 8, 3, 9, 10, 4, 5), Array(1, 2, 3, 4, 5))),
      6)

// Now we have an example DataFrame with columns for the predictions and targets
// We'll copy it a few times for different timestamps and then combine them into a single df to mimic backfill data
val df1 = predictionAndLabelsAndeScoresWithGroupsRDD.toDF("groups", "predictions", "labels").withColumn(timeColumn, lit(t1).cast(DataTypes.TimestampType))
val df2 = predictionAndLabelsAndeScoresWithGroupsRDD.toDF("groups", "predictions", "labels").withColumn(timeColumn, lit(t2).cast(DataTypes.TimestampType))
val df3 = predictionAndLabelsAndeScoresWithGroupsRDD.toDF("groups", "predictions", "labels").withColumn(timeColumn, lit(t3).cast(DataTypes.TimestampType))
val df = df1.union(df2).union(df3)
df.printSchema()

// Next we create a profiling session to compute RankingMetrics
// This must be a stand alone profiling session that does not compute
// other whylogs metrics. The default of k is 10 if not specified
val session = df.newProfilingSession("RankingMetricsTest") // start a new WhyLogs profiling job
  .withTimeColumn(timeColumn) // profiles generated for each unique time
  .withRankingMetrics(predictionField="predictions", targetField="labels", k=2)
  .groupBy("groups")

// COMMAND ----------

// Replace the following parameters below with your values after signing up for an account at https://whylabs.ai/
// You can find Organization Id on https://hub.whylabsapp.com/settings/access-tokens and the value looks something like: org-123abc
// also the settings page allows you t create new apiKeys which you will need an apiKey to upload to your account in Whylabs
// The modelId below specifies which model this profile is for, by default an initial model-1 is created but you will update this
// if you create a new model here https://hub.whylabsapp.com/settings/model-management
session.logRankingMetrics(
          orgId = "replace-with-org-id",
          modelId = "replace-with-model-id",
          apiKey = "replace-with-api-key")
