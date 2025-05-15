//package iceberg;
//
//import org.apache.iceberg.Schema;
//import org.apache.iceberg.catalog.Namespace;
//import org.apache.iceberg.catalog.TableIdentifier;
//import org.apache.iceberg.hadoop.HadoopCatalog;
//import org.apache.iceberg.types.Types;
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
//public class IcebergSparkExample3 {
//
//    public static void main(String[] args) throws NoSuchTableException {
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
//        /*spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg_catalog.iceberg_db");
//        spark.sql("USE iceberg_catalog.iceberg_db");
//        spark.sql("CREATE TABLE IF NOT EXISTS iceberg_catalog.iceberg_db.iceberg_table (id bigint, name string, age int) USING iceberg");
//        List<Row> data = Arrays.asList(
//                RowFactory.create(1L, "Alice", 30),
//                RowFactory.create(2L, "Bob", 25)
//        );
//        StructType schema = new StructType(new StructField[]{
//           new StructField("id", DataTypes.LongType, false, Metadata.empty()),
//           new StructField("name", DataTypes.StringType, false, Metadata.empty()),
//           new StructField("age", DataTypes.IntegerType, false, Metadata.empty())
//        });
//        spark.createDataFrame(data, schema)
//                .writeTo("iceberg_catalog.iceberg_db.iceberg_table")
//                .append();
//        Dataset<Row> users = spark.read()
//                .format("iceberg")
//                .load("iceberg_catalog.iceberg_db.iceberg_table");
//        users.show();*/
//
//        String warehouseLocation = "file:///tmp/iceberg/warehouse";
//        //String catalogName = "iceberg_catalog";
//        String dbName = "iceberg_catalog"; // In Iceberg terminology, a "namespace" is equivalent to what SQL traditionally calls a "database".
//        String tableName = "customer_data1";
//
//        HadoopCatalog catalog = new HadoopCatalog(spark.sparkContext().hadoopConfiguration(), warehouseLocation);
//
//        Namespace namespace = Namespace.of(dbName);
//
//        if (catalog.namespaceExists(namespace)) {
//            System.out.println("Namespace already exists");
//        } else {
//            catalog.createNamespace(namespace);
//            System.out.println("Namespace created successfully!");
//        }
//
//        TableIdentifier tableIdentifier = TableIdentifier.of(dbName, tableName);
//        Schema icebergSchema = new Schema(
//                Types.NestedField.required(1, "customer_id", Types.LongType.get()),
//                Types.NestedField.required(2, "name", Types.StringType.get()),
//                Types.NestedField.required(3, "email", Types.StringType.get()),
//                Types.NestedField.required(4, "signup_date", Types.DateType.get()),
//                Types.NestedField.required(5, "last_purchase", Types.TimestampType.withoutZone()),
//                Types.NestedField.required(6, "total_spent", Types.DoubleType.get())
//        );
//
//        if (catalog.tableExists(tableIdentifier)) {
//            System.out.println("Table already exists");
//        } else {
//            catalog.createTable(tableIdentifier, icebergSchema);
//            System.out.println("Namespace created successfully!");
//        }
//
//        catalog.listTables(Namespace.of("iceberg_catalog")).forEach(System.out::println);
//    }
//}