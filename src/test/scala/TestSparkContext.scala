import java.util.Date

import com.holdenkarau.spark.testing.SparkContextProvider
import org.apache.spark._
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.scalatest.{BeforeAndAfterAll, Suite}

import org.apache.spark.ml.feature.VectorAssembler

/** Shares a local `SparkContext` between all tests in a suite and closes it at the end. */
trait TestSparkContext extends BeforeAndAfterAll with SparkContextProvider {
  self: Suite =>

  @transient private var _sc: SparkContext = _
  @transient private var _sqlContext: SQLContext = _

  override def sc: SparkContext = _sc
  def sqlContext: SQLContext = _sqlContext

  val appID: String = new Date().toString + math.floor(math.random * 10E4).toLong.toString

  override val conf: SparkConf = new SparkConf().
    setMaster("local[*]").
    setAppName("test").
    set("spark.ui.enabled", "false").
    set("spark.app.id", appID)

  val spark: SparkSession = SparkSession.builder()
    .master("local[*]")
    .appName("test")
    .config("spark.ui.enabled", "false")
    .config("spark.app.id", appID)
    .getOrCreate()

  override def beforeAll() {
    _sc = spark.sparkContext
    _sqlContext = spark.sqlContext
    super.beforeAll()
  }

  def resourcePath(fileOrDirectory: String): String = {
    val currentDir = System.getProperty("user.dir")
    val resourcesPath = s"$currentDir/src/test/resources"
    s"$resourcesPath/$fileOrDirectory"
  }

  def transformedDF(df: DataFrame, features: Array[String]): DataFrame = {
    new VectorAssembler().setInputCols(features).setOutputCol("features").transform(df)
  }
}
