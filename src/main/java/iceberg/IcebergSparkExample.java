//package iceberg;
//
//import org.apache.iceberg.catalog.Catalog;
//import org.apache.iceberg.catalog.Namespace;
//import org.apache.iceberg.hadoop.HadoopCatalog;
//import org.apache.spark.sql.Dataset;
//import org.apache.spark.sql.Row;
//import org.apache.spark.sql.RowFactory;
//import org.apache.spark.sql.SparkSession;
//import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
//import org.apache.spark.sql.types.DataTypes;
//import org.apache.spark.sql.types.Metadata;
//import org.apache.spark.sql.types.StructField;
//import org.apache.spark.sql.types.StructType;
//
//import java.util.Arrays;
//import java.util.List;
//
//public class IcebergSparkExample {
//
//    public static void main(String[] args) throws NoSuchTableException {
//        SparkSession spark = SparkSession.builder()
//                .appName("Iceberg-Spark-Example")
//                .master("local[*]")
//                //.config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
//                .config("spark.sql.catalog.iceberg_catalog", "org.apache.iceberg.spark.SparkCatalog")
//                //.config("spark.sql.catalog.spark_catalog.type", "hive")
//                .config("spark.sql.catalog.iceberg_catalog.type", "hadoop")
//                //.config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
//                //.config("spark.sql.catalog.iceberg_catalog.warehouse", "hdfs://localhost:9870/user/iceberg/warehouse")
//                .config("spark.sql.catalog.iceberg_catalog.warehouse", "file:///tmp/iceberg/warehouse")
//                .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
//                .getOrCreate();
//
//        spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg_catalog.iceberg_db");
//        spark.sql("USE iceberg_catalog.iceberg_db");
//        spark.sql("CREATE TABLE IF NOT EXISTS iceberg_catalog.iceberg_db.iceberg_table (id bigint, name string, age int) USING iceberg");
//
//        List<Row> data = Arrays.asList(
//                RowFactory.create(1L, "Alice", 30),
//                RowFactory.create(2L, "Bob", 25)
//        );
//
//        StructType schema = new StructType(new StructField[]{
//           new StructField("id", DataTypes.LongType, false, Metadata.empty()),
//           new StructField("name", DataTypes.StringType, false, Metadata.empty()),
//           new StructField("age", DataTypes.IntegerType, false, Metadata.empty())
//        });
//
//        spark.createDataFrame(data, schema)
//                .writeTo("iceberg_catalog.iceberg_db.iceberg_table")
//                .append();
//
//        Dataset<Row> users = spark.read()
//                .format("iceberg")
//                .load("iceberg_catalog.iceberg_db.iceberg_table");
//
//        users.show();
//
//        String warehouseLocation = "file:///tmp/iceberg/warehouse";
//        HadoopCatalog catalog = new HadoopCatalog(spark.sparkContext().hadoopConfiguration(), warehouseLocation);
//
//        catalog.createNamespace(Namespace.of("iceberg_catalog"));
//        catalog.listTables(Namespace.of("iceberg_catalog")).forEach(System.out::println);
//    }
//}
