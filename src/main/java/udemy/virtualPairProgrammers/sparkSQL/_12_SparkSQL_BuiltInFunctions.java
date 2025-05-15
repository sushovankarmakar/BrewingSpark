package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class _12_SparkSQL_BuiltInFunctions {

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setAppName("spark-sql")
                .setMaster("local[*]");

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .getOrCreate();

        List<Row> inMemoryData = new ArrayList<>();
        inMemoryData.add(RowFactory.create("WARN", "2016-12-31 04:19:32"));
        inMemoryData.add(RowFactory.create("FATAL", "2016-12-31 03:22:34"));
        inMemoryData.add(RowFactory.create("WARN", "2016-12-31 03:21:21"));
        inMemoryData.add(RowFactory.create("INFO", "2015-4-21 14:32:21"));
        inMemoryData.add(RowFactory.create("FATAL", "2015-4-21 19:23:20"));

        StructType schema = new StructType(new StructField[]{
                new StructField("level", DataTypes.StringType, false, Metadata.empty()),
                new StructField("datetime", DataTypes.StringType, false, Metadata.empty())
        });

        Dataset<Row> inputData = sparkSession.createDataFrame(inMemoryData, schema);
        inputData.show(5);

        // ------------------------- Spark SQL built-in functions -------------------------
        inputData.createOrReplaceTempView("logging_table");
        Dataset<Row> groupedData = sparkSession.sql("SELECT level, count(datetime), collect_list(datetime) FROM logging_table GROUP BY level");
        groupedData.show(5, false);

        // ------------------------- date formatting, multiple grouping, ordering -------------------------
        // produce a report showing the number of FATAL, WARNING etc. for each month.
        Dataset<Row> dataByMonths = sparkSession.sql("SELECT" +
                        " level" +
                        ", date_format(datetime, 'MMMM') AS month" +
                        ", CAST(FIRST(date_format(datetime, 'M')) AS int) AS monthNum" +
                        ", COUNT(*) AS totalPerMonth" +
                        " FROM logging_table" +
                        " GROUP BY level, month" +
                        " ORDER BY monthNum ASC, level DESC")
                .drop("monthNum");

        // any column that is not part of the 'grouping' must have an Aggregation function performed on it.

        dataByMonths.show(200, false);

        Scanner sc = new Scanner(System.in);
        sc.nextLine();

        sparkSession.cloneSession();
    }
}
