import com.holdenkarau.spark.testing.LocalSparkContext
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, Suite}

trait SharedSparkContext extends BeforeAndAfterAll {
  self: Suite =>

  @transient private var _sc: SparkContext = _
//  @transient private var _ss: SparkSession = _

  implicit def sc: SparkContext = _sc
//  implicit def ss: SparkSession = _ss

  val conf: SparkConf = new SparkConf().setMaster("local[*]")
    .setAppName("test")

  override def beforeAll() {
//    _ss = new SparkSession(conf)
//    _sc = _ss.sparkContext
    _sc = new SparkContext(conf)
    super.beforeAll()
  }

  override def afterAll() {
    LocalSparkContext.stop(_sc)
    _sc = null
    super.afterAll()
  }
}
