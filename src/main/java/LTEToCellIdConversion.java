import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;

public class LTEToCellIdConversion {

    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder()
                .appName("LTEToCellIdConversion")
                .master("local")
                .getOrCreate();

        // Sample data
        Dataset<Row> df = spark.createDataFrame(java.util.Arrays.asList(
                new CellId("123456789"),
                new CellId("987654321"),
                new CellId("112233445566")
        ), CellId.class);

        int minLength = 8;

        long startTime = System.currentTimeMillis();

        Dataset<Row> dfConverted = df.withColumn("hexaPart", lpad(expr("cast(conv(cell_id, 10, 16) as string)"), minLength, "0"))
                .withColumn("enodeHex", expr("substring(hexaPart, 1, 6)"))
                .withColumn("cidHex", expr("substring(hexaPart, length(hexaPart) - 1, 2)"))
                .withColumn("enode", expr("conv(enodeHex, 16, 10)"))
                .withColumn("cid", expr("conv(cidHex, 16, 10)"))
                .withColumn("converted_cell_id", concat_ws("-", col("enode"), col("cid")))
                .select("cell_id", "converted_cell_id");

        dfConverted.show();

        long endTime = System.currentTimeMillis();
        System.out.println("Time taken: " + (endTime - startTime) + " ms");

        spark.stop();
    }

    // Class for sample data
    public static class CellId implements java.io.Serializable {
        private String cell_id;

        public CellId() {
        }

        public CellId(String cell_id) {
            this.cell_id = cell_id;
        }

        public String getCell_id() {
            return cell_id;
        }

        public void setCell_id(String cell_id) {
            this.cell_id = cell_id;
        }
    }
}
