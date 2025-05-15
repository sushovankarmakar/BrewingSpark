//package iceberg;
//
//import lombok.extern.slf4j.Slf4j;
//import org.apache.spark.sql.*;
//import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
//
//@Slf4j
//public class IcebergSparkExample2 {
//
//    public static void main(String[] args) {
//
//        SparkSession spark = SparkSession.builder()
//                .appName("Iceberg-Spark-Example")
//                .master("local[*]")
//                .config("spark.sql.catalog.iceberg_catalog", "org.apache.iceberg.spark.SparkCatalog")
//                .config("spark.sql.catalog.iceberg_catalog.type", "hadoop")
//                .config("spark.sql.catalog.iceberg_catalog.warehouse", "file:///tmp/iceberg/warehouse")
//                .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
//                .getOrCreate();
//
//        String catalogName = "iceberg_catalog";
//        String dbName = "customer_db";
//        String tableName = "customer_data";
//        String tableIdentifier = catalogName + "." + dbName + "." + tableName;
//
//        try {
//
//            spark.sql("CREATE DATABASE IF NOT EXISTS " + catalogName + "." + dbName);
//
//            /*StructType schema = DataTypes.createStructType(new StructField[]{
//                    DataTypes.createStructField("customer_id", DataTypes.LongType, false),
//                    DataTypes.createStructField("name", DataTypes.StringType, false),
//                    DataTypes.createStructField("email", DataTypes.StringType, false),
//                    DataTypes.createStructField("signup_date", DataTypes.StringType, false),
//                    DataTypes.createStructField("last_purchase", DataTypes.TimestampType, true),
//                    DataTypes.createStructField("total_spent", DataTypes.DoubleType, true)
//            });*/
//
//            String createTableSQL = String.format(
//                    "CREATE TABLE IF NOT EXISTS %s (" +
//                            "customer_id BIGINT, " +
//                            "name STRING, " +
//                            "email STRING, " +
//                            "signup_date DATE, " +
//                            "last_purchase TIMESTAMP, " +
//                            "total_spent DOUBLE) " +
//                            "USING iceberg " +
//                            "PARTITIONED BY (days(signup_date))",
//                    tableIdentifier
//            );
//            spark.sql(createTableSQL);
//
//            insertSampleData(spark, tableIdentifier);
//
//            queryData(spark, tableIdentifier);
//
//        } catch (NoSuchTableException e) {
//            throw new RuntimeException(e);
//        } finally {
//            spark.close();
//        }
//    }
//
//    private static void insertSampleData(SparkSession spark, String tableIdentifier) throws NoSuchTableException {
//        // Generate sample data using DataFrame API
//        Dataset<Row> data = spark.sql("SELECT 1 as customer_id, 'John Doe' as name, " +
//                "'john@example.com' as email, CAST('2023-01-15' AS DATE) as signup_date, " +
//                "CURRENT_TIMESTAMP() as last_purchase, 245.50 as total_spent");
//
//        // Append more rows
//        data = data.union(
//                spark.sql("SELECT 2 as customer_id, 'Jane Smith' as name, " +
//                        "'jane@example.com' as email, CAST('2023-02-20' AS DATE) as signup_date, " +
//                        "CURRENT_TIMESTAMP() as last_purchase, 352.75 as total_spent")
//        ).union(
//                spark.sql("SELECT 3 as customer_id, 'Bob Johnson' as name, " +
//                        "'bob@example.com' as email, CAST('2023-03-10' AS DATE) as signup_date, " +
//                        "CURRENT_TIMESTAMP() as last_purchase, 128.99 as total_spent")
//        );
//
//        // Write data to Iceberg table
//        data.writeTo(tableIdentifier).append();
//
//        System.out.println("Sample data inserted successfully!");
//    }
//
//    private static void queryData(SparkSession spark, String tableIdentifier) {
//
//        log.info("All customers :");
//        spark.read().format("iceberg")
//                .load(tableIdentifier)
//                .show(false);
//
//        log.info("\nCustomers who spent more than $200:");
//        spark.read().format("iceberg")
//                .load(tableIdentifier)
//                .filter(functions.col("total_spent").gt(200))
//                .show(false);
//
//        log.info("\nTotal spent by signup month:");
//        spark.read().format("iceberg")
//                .load(tableIdentifier)
//                .withColumn("signup_month", functions.month(functions.col("signup_date")))
//                .groupBy("signup_month")
//                .agg(functions.sum("total_spent").as("total_spent"),
//                        functions.count("*").as("customer_count"))
//                .orderBy("signup_month")
//                .show(false);
//    }
//
//}